"""netexec builder + parser unit tests (CORE-04): quoting, domain form, case-insensitive prefix, whitespace.

Pure functions are tested directly; the service is exercised Docker-free against FakeExegolAdapter.
"""

import inspect
import os
import re
import shlex
import subprocess
from types import SimpleNamespace

import pytest
from mcp import Client

from exegol_mcp.app import build_server
from exegol_mcp.models.exec import ExecResult
from exegol_mcp.services.remote import (
    VALID_PROTOCOLS,
    _build_netexec_command,
    _parse_netexec_output,
    _remove_netexec_prefix,
    exec_remote_command,
)
from exegol_mcp.tools import remote as remote_tool
from tests.fakes import FakeExegolAdapter


def _stopped(adapter: FakeExegolAdapter, name: str) -> FakeExegolAdapter:
    adapter.running[name] = False
    return adapter


# --- _build_netexec_command -------------------------------------------------------------------------------


def test_build_uses_nxc_protocol_target_and_plain_user_without_domain() -> None:
    cmd = _build_netexec_command("ssh", "10.0.0.1", "root", "pw", "id")
    assert cmd.startswith("nxc ssh 10.0.0.1 ")
    assert "-u root" in cmd


def test_build_sends_domain_backslash_user_when_a_domain_is_given() -> None:
    # SEC-01: the assembled -u value is now shlex-quoted, so the DOMAIN\user form is wrapped (the backslash
    # forces quoting). The quoting preserves the backslash through the single eval (see the real-boundary test).
    cmd = _build_netexec_command("smb", "10.0.0.1", "admin", "pw", "id", domain="CORP")
    quoted_user = shlex.quote("CORP\\admin")
    assert f"-u {quoted_user}" in cmd
    assert "-u 'CORP\\admin'" in cmd


def test_build_quotes_the_password_with_shlex() -> None:
    cmd = _build_netexec_command("smb", "10.0.0.1", "admin", "p@ss word!", "id")
    assert f"-p {shlex.quote('p@ss word!')}" in cmd
    assert "-p 'p@ss word!'" in cmd


def test_build_quotes_the_command_with_shlex() -> None:
    cmd = _build_netexec_command("smb", "10.0.0.1", "admin", "pw", "whoami && id")
    assert f"-x {shlex.quote('whoami && id')}" in cmd


def test_build_passes_the_port_as_a_separate_port_option() -> None:
    # G-03-2: netexec does not read a port from the target string (an IP:PORT target silently does nothing), so
    # the target is emitted alone and the port goes through netexec's own --port option.
    cmd = _build_netexec_command("ssh", "10.0.0.1", "root", "pw", "id", port=2222)
    assert cmd.startswith("nxc ssh 10.0.0.1 --port 2222 -u ")
    assert "10.0.0.1:2222" not in cmd


def test_build_without_a_port_emits_no_port_option() -> None:
    cmd = _build_netexec_command("ssh", "10.0.0.1", "root", "pw", "id")
    assert "--port" not in cmd
    assert cmd == "nxc ssh 10.0.0.1 -u root -p pw -x id"


def test_build_uses_capital_x_flag_for_powershell() -> None:
    cmd = _build_netexec_command("winrm", "10.0.0.1", "admin", "pw", "Get-Host", use_powershell=True)
    assert " -X " in cmd
    assert " -x " not in cmd


# --- SEC-01: drive the built string through the REAL exec.py boundary (one `eval "$CMD"`) -----------------
#
# adapter/exec.py hands the built string to the container as the CMD env var and the SDK runs it with a single
# `eval "$CMD"`. These tests reproduce that exact boundary: define an `nxc` shell function, set CMD to the built
# string and `eval "$CMD"`, so an unquoted field that could add a command is caught end to end rather than by a
# substring assertion. A real PATH is provided so that if injection DID work, `touch` would be found and run.


def _eval_built(built: str, nxc_body: str = ":") -> subprocess.CompletedProcess[str]:
    """Run the built netexec string through the real single-eval boundary with a stubbed `nxc`."""
    script = f'nxc() {{ {nxc_body}\n}}\neval "$CMD"'
    return subprocess.run(
        ["/bin/sh", "-c", script],
        env={"CMD": built, "PATH": os.environ.get("PATH", "/usr/bin:/bin")},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("field", ["target_ip", "username", "domain", "password", "command"])
@pytest.mark.parametrize(
    "payload_tmpl",
    ["x; touch {s}", "x$(touch {s})", "x`touch {s}`", "x\"; touch {s} ;\"", "x'; touch {s} ;'"],
)
def test_build_blocks_injection_in_every_field_through_the_real_eval_boundary(
    tmp_path, field: str, payload_tmpl: str
) -> None:
    sentinel = tmp_path / "pwned"
    payload = payload_tmpl.format(s=sentinel)
    kwargs: dict[str, object] = {
        "protocol": "ssh",
        "target_ip": "10.0.0.1",
        "username": "root",
        "password": "pw",
        "command": "id",
    }
    if field == "domain":
        built = _build_netexec_command(**kwargs, domain=payload)  # type: ignore[arg-type]
    else:
        kwargs[field] = payload
        built = _build_netexec_command(**kwargs)  # type: ignore[arg-type]
    result = _eval_built(built)
    assert not sentinel.exists(), (
        f"injection via {field} created the sentinel; built={built!r} stderr={result.stderr!r}"
    )


def test_build_preserves_domain_backslash_user_through_the_real_eval_boundary() -> None:
    built = _build_netexec_command("smb", "10.0.0.1", "admin", "pw", "id", domain="CORP")
    result = _eval_built(built, nxc_body='printf "%s\\n" "$@"')
    assert "CORP\\admin" in result.stdout.splitlines()


def test_build_plain_user_without_domain_reaches_nxc_verbatim_through_the_real_eval_boundary() -> None:
    built = _build_netexec_command("ssh", "10.0.0.1", "admin", "pw", "id")
    result = _eval_built(built, nxc_body='printf "%s\\n" "$@"')
    assert "admin" in result.stdout.splitlines()


def test_build_password_with_spaces_reaches_nxc_as_a_single_argument() -> None:
    built = _build_netexec_command("ssh", "10.0.0.1", "root", "p@ss word!", "id")
    result = _eval_built(built, nxc_body='printf "%s\\n" "$@"')
    assert "p@ss word!" in result.stdout.splitlines()


def test_build_port_reaches_nxc_as_its_own_argv_words_through_the_real_eval_boundary() -> None:
    built = _build_netexec_command("ssh", "10.0.0.1", "root", "pw", "id", port=2222)
    result = _eval_built(built, nxc_body='printf "%s\\n" "$@"')
    assert result.stdout.splitlines() == ["ssh", "10.0.0.1", "--port", "2222", "-u", "root", "-p", "pw", "-x", "id"]


# --- _remove_netexec_prefix: the case fix and the whitespace fix ------------------------------------------


def test_prefix_is_stripped_case_insensitively() -> None:
    # netexec prints the protocol uppercase; v1 compared parts[0].upper() against a lowercase list and never
    # matched, so the prefix was never stripped. It must be stripped now.
    line = "SSH         10.0.0.1        22     10.0.0.1         uid=0(root)"
    assert _remove_netexec_prefix(line) == "uid=0(root)"


def test_prefix_stripping_preserves_internal_whitespace_in_the_output() -> None:
    # v1 rebuilt the remainder with ' '.join(split tokens), collapsing runs of spaces. The content spacing
    # must survive.
    line = "SMB    10.0.0.1    445    10.0.0.1    col1     col2        col3"
    assert _remove_netexec_prefix(line) == "col1     col2        col3"


def test_a_line_without_a_recognisable_prefix_is_returned_unchanged() -> None:
    assert _remove_netexec_prefix("plain output line") == "plain output line"


def test_parse_extracts_the_output_after_the_executed_command_marker() -> None:
    # netexec prefixes each line with "PROTOCOL  IP  PORT  IP"; the output follows the Executed-command marker.
    raw = (
        "SMB  10.0.0.1  445  10.0.0.1  [*] Windows 10\n"
        "SMB  10.0.0.1  445  10.0.0.1  [+] CORP\\admin:pw\n"
        "SMB  10.0.0.1  445  10.0.0.1  [+] Executed command\n"
        "SMB  10.0.0.1  445  10.0.0.1  hello     world\n"
    )
    assert _parse_netexec_output(raw) == "hello     world"


# Successful single-target runs whose log lines before the marker contain `[-]` (WR-06). The host logged in and ran
# the command, so only the command output comes back, as at fb938e7. Hand-built from netexec's line format.
WMIEXEC_FALLBACK_DIAGNOSTIC = (
    "[-] WMIEXEC: Could not retrieve output file, it may have been detected by AV. Retrying with atexec"
)
SINGLE_TARGET_SUCCESS_RUNS = [
    pytest.param(
        (
            "SMB  10.0.0.1  445  DC01  [*] Windows Server 2019 (name:DC01) (domain:corp)\n"
            "SMB  10.0.0.1  445  DC01  [+] corp\\admin:pw (Pwn3d!)\n"
            f"SMB  10.0.0.1  445  DC01  {WMIEXEC_FALLBACK_DIAGNOSTIC}\n"
            "SMB  10.0.0.1  445  DC01  [+] Executed command via atexec\n"
            "SMB  10.0.0.1  445  DC01  corp\\admin\n"
        ),
        "SMB  10.0.0.1  445  DC01  corp\\admin",
        id="exec-method-fallback-line",
    ),
    pytest.param(
        (
            "SMB  10.0.0.1  445  DC01  [*] Windows Server 2019 (name:DC01) (domain:corp)\n"
            "SMB  10.0.0.1  445  DC01  [+] corp\\admin:p[-]ss (Pwn3d!)\n"
            "SMB  10.0.0.1  445  DC01  [+] Executed command via wmiexec\n"
            "SMB  10.0.0.1  445  DC01  corp\\admin\n"
        ),
        "SMB  10.0.0.1  445  DC01  corp\\admin",
        id="dash-marker-in-the-password",
    ),
    pytest.param(
        (
            "SMB  10.0.0.1  445  DC01  [*] Windows Server 2019 (name:DC01) (domain:c[-]rp)\n"
            "SMB  10.0.0.1  445  DC01  [+] c[-]rp\\admin:pw (Pwn3d!)\n"
            "SMB  10.0.0.1  445  DC01  [+] Executed command via wmiexec\n"
            "SMB  10.0.0.1  445  DC01  out\n"
        ),
        "SMB  10.0.0.1  445  DC01  out",
        id="dash-marker-in-a-target-controlled-domain",
    ),
]


@pytest.mark.parametrize(("raw", "expected"), SINGLE_TARGET_SUCCESS_RUNS)
def test_parse_of_a_single_target_successful_run_keeps_only_the_command_output(raw: str, expected: str) -> None:
    # WR-06: a `[-]` line before the marker from the host that then ran the command is a diagnostic, not a rejected
    # login.
    assert _parse_netexec_output(raw) == expected


# A two-host SMB run (target 10.0.0.0/30) where DC01 rejects the login before WS01's `[+] Executed command`. netexec
# runs hosts concurrently, so a rejection printing before another host's marker is a normal ordering (G-03-4, WR-04).
# Hand-built from netexec's line format, as in the verifier's wire reproduction; not captured live.
TWO_HOST_SMB_RUN = (
    "SMB  10.0.0.1  445  DC01  [*] Windows Server 2019 (name:DC01) (domain:corp)\n"
    "SMB  10.0.0.2  445  WS01  [*] Windows 10 (name:WS01) (domain:corp)\n"
    "SMB  10.0.0.1  445  DC01  [-] corp\\admin:pw STATUS_LOGON_FAILURE\n"
    "SMB  10.0.0.2  445  WS01  [+] corp\\admin:pw (Pwn3d!)\n"
    "SMB  10.0.0.2  445  WS01  [+] Executed command via wmiexec\n"
    "SMB  10.0.0.2  445  WS01  corp\\admin\n"
)

# A two-host SSH run (target 10.0.0.0/30) where 10.0.0.1 rejects and 10.0.0.2 runs `id` (WR-07). netexec puts the IP
# in column 4 on SSH, so stripping the prefix would leave no host on the failure line. Hand-built from netexec's line
# format.
TWO_HOST_SSH_RUN = (
    "SSH  10.0.0.1  22  10.0.0.1  [*] SSH-2.0-OpenSSH_9.2p1\n"
    "SSH  10.0.0.2  22  10.0.0.2  [*] SSH-2.0-OpenSSH_9.2p1\n"
    "SSH  10.0.0.1  22  10.0.0.1  [-] root:pw\n"
    "SSH  10.0.0.2  22  10.0.0.2  [+] root:pw  Linux - Shell access!\n"
    "SSH  10.0.0.2  22  10.0.0.2  [+] Executed command\n"
    "SSH  10.0.0.2  22  10.0.0.2  uid=0(root)\n"
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            (
                "SSH  10.0.0.1  22  10.0.0.1  [*] SSH-2.0-OpenSSH_9.2p1\n"
                "SSH  10.0.0.1  22  10.0.0.1  [-] uatuser:pw\n"
            ),
            "SSH-2.0-OpenSSH_9.2p1\n[-] uatuser:pw",
        ),
        *[
            (
                (
                    f"{proto}  10.0.0.1  {port}  DC01  [*] banner (name:DC01) (domain:corp.local)\n"
                    f"{proto}  10.0.0.1  {port}  DC01  [-] corp.local\\admin:pw STATUS_LOGON_FAILURE\n"
                ),
                (
                    "banner (name:DC01) (domain:corp.local)\n"
                    f"{proto}  10.0.0.1  {port}  DC01  [-] corp.local\\admin:pw STATUS_LOGON_FAILURE"
                ),
            )
            for proto, port in (
                ("SMB", 445), ("WINRM", 5985), ("MSSQL", 1433), ("WMI", 135), ("RDP", 3389), ("SSH", 22)
            )
        ],
    ],
    ids=["ssh-ip-in-column-4", "smb-dc01", "winrm-dc01", "mssql-dc01", "wmi-dc01", "rdp-dc01", "ssh-dc01"],
)
def test_parse_of_a_single_target_failed_login_is_unchanged(raw: str, expected: str) -> None:
    # Characterization (G-03-4): a single-target failed login has no `[+] Executed command` marker, so it goes
    # through the fallback. Its parsed output is pinned byte-for-byte to what it was before the main loop started
    # keeping `[-]` lines (fb938e7).
    assert _parse_netexec_output(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            TWO_HOST_SMB_RUN,
            (
                "SMB  10.0.0.1  445  DC01  [-] corp\\admin:pw STATUS_LOGON_FAILURE\n"
                "SMB  10.0.0.2  445  WS01  corp\\admin"
            ),
        ),
        (
            (
                "SMB  10.0.0.2  445  WS01  [*] Windows 10 (name:WS01) (domain:corp)\n"
                "SMB  10.0.0.2  445  WS01  [+] corp\\admin:pw (Pwn3d!)\n"
                "SMB  10.0.0.2  445  WS01  [+] Executed command via wmiexec\n"
                "SMB  10.0.0.2  445  WS01  corp\\admin\n"
                "SMB  10.0.0.1  445  DC01  [-] corp\\admin:pw STATUS_LOGON_FAILURE\n"
            ),
            (
                "SMB  10.0.0.2  445  WS01  corp\\admin\n"
                "SMB  10.0.0.1  445  DC01  [-] corp\\admin:pw STATUS_LOGON_FAILURE"
            ),
        ),
        (
            (
                "SMB  10.0.0.2  445  WS01  [+] Executed command via wmiexec\n"
                "SMB  10.0.0.2  445  WS01  corp\\admin\n"
                "SMB  10.0.0.1  445  DC01  [-] corp\\admin:x[+] Executed command STATUS_LOGON_FAILURE\n"
                "SMB  10.0.0.3  445  WS02  ws02-out\n"
            ),
            (
                "SMB  10.0.0.2  445  WS01  corp\\admin\n"
                "SMB  10.0.0.1  445  DC01  [-] corp\\admin:x[+] Executed command STATUS_LOGON_FAILURE\n"
                "SMB  10.0.0.3  445  WS02  ws02-out"
            ),
        ),
        (
            (
                "SMB  10.0.0.1  445  DC01  [*] Windows Server 2019 (name:DC01) (domain:corp)\n"
                "SMB  10.0.0.2  445  WS01  [*] Windows 10 (name:WS01) (domain:corp)\n"
                "SMB  10.0.0.1  445  DC01  [-] corp\\admin:pw STATUS_LOGON_FAILURE\n"
                "SMB  10.0.0.2  445  WS01  [+] corp\\admin:pw (Pwn3d!)\n"
                f"SMB  10.0.0.2  445  WS01  {WMIEXEC_FALLBACK_DIAGNOSTIC}\n"
                "SMB  10.0.0.2  445  WS01  [+] Executed command via atexec\n"
                "SMB  10.0.0.2  445  WS01  corp\\admin\n"
            ),
            (
                "SMB  10.0.0.1  445  DC01  [-] corp\\admin:pw STATUS_LOGON_FAILURE\n"
                "SMB  10.0.0.2  445  WS01  corp\\admin"
            ),
        ),
        (
            (
                "SMB  10.0.0.1  445  DC01  [+] Executed command via wmiexec\n"
                "SMB  10.0.0.1  445  DC01  dc01-out\n"
                f"SMB  10.0.0.2  445  WS01  {WMIEXEC_FALLBACK_DIAGNOSTIC}\n"
                "SMB  10.0.0.2  445  WS01  [+] Executed command via atexec\n"
                "SMB  10.0.0.2  445  WS01  ws01-out\n"
            ),
            (
                "SMB  10.0.0.1  445  DC01  dc01-out\n"
                "SMB  10.0.0.2  445  WS01  ws01-out"
            ),
        ),
        (
            (
                "SMB  10.0.0.1  445  DC01  [-] corp\\admin:pw STATUS_LOGON_FAILURE\n"
                "SMB  10.0.0.2  445  DC01  [+] Executed command via wmiexec\n"
                "SMB  10.0.0.2  445  DC01  dc01-out\n"
            ),
            (
                "SMB  10.0.0.1  445  DC01  [-] corp\\admin:pw STATUS_LOGON_FAILURE\n"
                "SMB  10.0.0.2  445  DC01  dc01-out"
            ),
        ),
        (
            TWO_HOST_SSH_RUN,
            (
                "SSH  10.0.0.1  22  10.0.0.1  [-] root:pw\n"
                "uid=0(root)"
            ),
        ),
        (
            (
                "SMB  10.0.0.1  445  1.2.3.²  [-] corp\\admin:pw STATUS_LOGON_FAILURE\n"
                "SMB  10.0.0.2  445  WS01  [+] Executed command via wmiexec\n"
                "SMB  10.0.0.2  445  WS01  corp\\admin\n"
            ),
            (
                "SMB  10.0.0.1  445  1.2.3.²  [-] corp\\admin:pw STATUS_LOGON_FAILURE\n"
                "SMB  10.0.0.2  445  WS01  corp\\admin"
            ),
        ),
    ],
    ids=[
        "failure-before-the-other-hosts-marker",
        "failure-after-the-other-hosts-output",
        "marker-text-credential-after-the-marker",
        "a-diagnostic-of-a-host-that-ran-is-dropped-next-to-a-rejection",
        "a-diagnostic-after-another-hosts-output-is-dropped-when-its-host-runs",
        "a-host-reusing-the-rejecting-hosts-name-cannot-hide-its-rejection",
        "ssh-failure-before-the-other-hosts-marker",
        "a-unicode-digit-name-on-a-kept-failure-line",
    ],
)
def test_parse_keeps_a_failure_line_wherever_it_sits_in_a_multi_target_run(raw: str, expected: str) -> None:
    # G-03-4/WR-04: netexec runs hosts concurrently, so a failure can sit before or after another host's output, and
    # every `[-]` line keeps its place.
    assert _parse_netexec_output(raw) == expected


def test_parse_keeps_a_failure_line_when_the_other_hosts_command_printed_nothing() -> None:
    # G-03-4: the marker is present but no command output follows, so the main loop collects nothing and the fallback
    # rescans every line. Containment only: the fallback's banner rescan (WR-02, open) must not be pinned here.
    raw = TWO_HOST_SMB_RUN.removesuffix("SMB  10.0.0.2  445  WS01  corp\\admin\n")
    assert raw != TWO_HOST_SMB_RUN
    parsed = _parse_netexec_output(raw).splitlines()
    assert "SMB  10.0.0.1  445  DC01  [-] corp\\admin:pw STATUS_LOGON_FAILURE" in parsed


@pytest.mark.parametrize(
    ("raw", "expected_failures"),
    [
        (
            (
                "SSH  10.0.0.1  22  10.0.0.1  [*] SSH-2.0-OpenSSH_9.2p1\n"
                "SSH  10.0.0.2  22  10.0.0.2  [*] SSH-2.0-OpenSSH_9.2p1\n"
                "SSH  10.0.0.1  22  10.0.0.1  [-] root:pw\n"
                "SSH  10.0.0.2  22  10.0.0.2  [-] root:pw\n"
            ),
            ["SSH  10.0.0.1  22  10.0.0.1  [-] root:pw", "SSH  10.0.0.2  22  10.0.0.2  [-] root:pw"],
        ),
        (
            (
                "SMB  10.0.0.1  445  DC01  [*] Windows Server 2019 (name:DC01) (domain:corp)\n"
                "SMB  10.0.0.2  445  WS01  [*] Windows 10 (name:WS01) (domain:corp)\n"
                "SMB  10.0.0.1  445  DC01  [-] corp\\admin:pw STATUS_LOGON_FAILURE\n"
                "SMB  10.0.0.2  445  WS01  [-] corp\\admin:pw STATUS_LOGON_FAILURE\n"
            ),
            [
                "SMB  10.0.0.1  445  DC01  [-] corp\\admin:pw STATUS_LOGON_FAILURE",
                "SMB  10.0.0.2  445  WS01  [-] corp\\admin:pw STATUS_LOGON_FAILURE",
            ],
        ),
    ],
    ids=["ssh-ip-in-column-4", "smb-netbios-name-in-column-4"],
)
def test_parse_names_every_rejecting_host_when_no_host_ran_the_command(raw: str, expected_failures: list[str]) -> None:
    # WR-07: with no marker the fallback runs. Each kept `[-]` line keeps its netexec prefix in a multi-target run.
    # Only the `[-]` lines are compared, so the fallback's banner rescan is not pinned here.
    assert [line for line in _parse_netexec_output(raw).splitlines() if "[-]" in line] == expected_failures


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        *[
            (
                (
                    "SSH  10.0.0.1  22  10.0.0.1  [*] SSH-2.0-OpenSSH_9.2p1\n"
                    f"SSH  10.0.0.1  22  10.0.0.1  [-] {cred}\n"
                ),
                f"[-] {cred}",
            )
            for cred in ("uatuser:p[+]ss", "uatuser:a[*]b", "ad[+]min:pw", "uatuser:x[+] Executed command")
        ],
        (
            (
                "SMB  10.0.0.1  445  DC01  [*] Windows Server 2019 Build 17763 x64 (name:DC01) (domain:corp.local)\n"
                "SMB  10.0.0.1  445  DC01  [-] corp.local\\admin:p[+]ss STATUS_LOGON_FAILURE\n"
            ),
            "[-] corp.local\\admin:p[+]ss STATUS_LOGON_FAILURE",
        ),
        (
            (
                "SSH  10.0.0.1  22  10.0.0.1  [*] SSH-2.0-OpenSSH_9.2p1\n"
                "SSH  10.0.0.1  22  10.0.0.1  [-] uatuser:x[+] Executed command\n"
                "Running nxc against 1 target\n"
            ),
            "[-] uatuser:x[+] Executed command",
        ),
    ],
    ids=["ssh-plus-in-password", "ssh-star-in-password", "ssh-plus-in-username", "ssh-marker-text-in-password",
         "smb-netbios-plus-in-password", "ssh-marker-text-in-password-then-a-trailing-line"],
)
def test_parse_never_cuts_a_failure_line_whose_credential_contains_a_status_marker(raw: str, expected: str) -> None:
    # G-03-4: netexec exits 0 on an authentication failure, so the `[-]` line is the agent's only failure signal. A
    # `[+]`/`[*]` inside the credential or the backend's error text must not slice the marker off that line.
    assert expected in _parse_netexec_output(raw)


# --- exec_remote_command service --------------------------------------------------------------------------


async def test_remote_rejects_an_invalid_protocol_naming_the_valid_set() -> None:
    adapter = FakeExegolAdapter()
    with pytest.raises(ValueError, match="Invalid protocol 'telnet'") as raised:
        await exec_remote_command(adapter, "exegol-lab", "telnet", "10.0.0.1", "root", "pw", "id")
    for protocol in VALID_PROTOCOLS:
        assert protocol in str(raised.value)


async def test_remote_accepts_a_protocol_regardless_of_case() -> None:
    adapter = FakeExegolAdapter(exec_router=lambda _cmd: ExecResult(stdout="ok", stderr="", exit_code=0))
    result = await exec_remote_command(adapter, "exegol-lab", "SSH", "10.0.0.1", "root", "pw", "id")
    assert result.exit_code == 0
    assert adapter.exec_commands and adapter.exec_commands[0].startswith("nxc ssh 10.0.0.1 ")


@pytest.mark.parametrize("port", [0, -1, 65536])
async def test_remote_rejects_an_out_of_range_port_before_touching_the_container(port: int) -> None:
    adapter = FakeExegolAdapter(exec_router=lambda _cmd: ExecResult(stdout="ok", stderr="", exit_code=0))
    with pytest.raises(ValueError, match="port") as raised:
        await exec_remote_command(adapter, "exegol-lab", "ssh", "10.0.0.1", "root", "pw", "id", port=port)
    assert "65535" in str(raised.value)
    assert adapter.exec_commands == []  # validation happens before any command is built or run


@pytest.mark.parametrize("port", [1, 65535])
async def test_remote_accepts_the_port_range_bounds(port: int) -> None:
    adapter = FakeExegolAdapter(exec_router=lambda _cmd: ExecResult(stdout="ok", stderr="", exit_code=0))
    result = await exec_remote_command(adapter, "exegol-lab", "ssh", "10.0.0.1", "root", "pw", "id", port=port)
    assert result.exit_code == 0
    assert adapter.exec_commands and adapter.exec_commands[0].startswith(f"nxc ssh 10.0.0.1 --port {port} -u ")


async def test_the_remote_tool_sends_the_port_to_the_backend_as_a_port_option() -> None:
    # End to end from the tool handler: with no job store attached the service runs the inline foreground path,
    # so the adapter records the exact command the backend would evaluate.
    adapter = FakeExegolAdapter(exec_router=lambda _cmd: ExecResult(stdout="ok", stderr="", exit_code=0))
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=adapter))
    await remote_tool.exec_remote_command(
        container_name="exegol-lab",
        protocol="ssh",
        target_ip="10.0.0.1",
        username="root",
        password="pw",
        command="id",
        port=2222,
        ctx=ctx,  # type: ignore[arg-type]
    )
    assert adapter.exec_commands and adapter.exec_commands[0].startswith("nxc ssh 10.0.0.1 --port 2222 ")


async def test_remote_unknown_container_raises_value_error() -> None:
    with pytest.raises(ValueError, match="not found"):
        await exec_remote_command(FakeExegolAdapter(), "ghost", "ssh", "10.0.0.1", "root", "pw", "id")


async def test_remote_stopped_container_raises_start_it_first() -> None:
    adapter = _stopped(FakeExegolAdapter(), "exegol-lab")
    with pytest.raises(RuntimeError, match="start it first"):
        await exec_remote_command(adapter, "exegol-lab", "ssh", "10.0.0.1", "root", "pw", "id")


async def test_remote_backend_failure_is_a_result_with_the_backend_exit_code_not_an_exception() -> None:
    unreachable = ExecResult(stdout="SSH  10.0.0.1  22  HOST  [-] Connection refused", stderr="", exit_code=1)
    adapter = FakeExegolAdapter(exec_router=lambda _cmd: unreachable)
    result = await exec_remote_command(adapter, "exegol-lab", "ssh", "10.0.0.1", "root", "pw", "id")
    assert result.exit_code == 1


async def test_remote_auth_failure_keeps_the_backend_exit_code_and_shows_the_failure_in_the_output() -> None:
    # G-03-3: netexec exits 0 when authentication fails (the live UAT result). The service passes that code through
    # unchanged (EXEC-08: no code is synthesized from the parsed output) and the failure is visible in the output as
    # a line containing `[-]`, returned unredacted (D-09). Here it reads `[-] uatuser:pw` because netexec put the IP
    # in column 4 and the prefix is stripped; other protocols and host names keep the prefix in front of the marker
    # (see the per-protocol test below).
    raw = (
        "SSH  10.0.0.1  22  10.0.0.1  [*] SSH-2.0-OpenSSH_9.2p1\n"
        "SSH  10.0.0.1  22  10.0.0.1  [-] uatuser:pw\n"
    )
    adapter = FakeExegolAdapter(exec_router=lambda _cmd: ExecResult(stdout=raw, stderr="", exit_code=0))
    result = await exec_remote_command(adapter, "exegol-lab", "ssh", "10.0.0.1", "uatuser", "pw", "id")
    assert result.exit_code == 0
    assert "[-] uatuser:pw" in result.stdout


@pytest.mark.parametrize(
    ("proto", "port"),
    [("SMB", 445), ("WINRM", 5985), ("MSSQL", 1433), ("WMI", 135), ("RDP", 3389), ("SSH", 22)],
)
async def test_remote_auth_failure_with_a_host_name_in_column_4_keeps_exit_0_and_a_line_containing_the_marker(
    proto: str, port: int
) -> None:
    # G-03-4: on the Windows protocols (and SSH to a host name) netexec puts a NetBIOS/host name in column 4, so the
    # prefix is not stripped and the `[-]` marker sits after it. The failure must still be a line containing `[-]`
    # with the credential text; where the marker sits in the line is deliberately not asserted. Fixtures are
    # hand-built from netexec's line format (only SSH was captured live).
    raw = (
        f"{proto}  10.0.0.1  {port}  DC01  [*] banner (name:DC01) (domain:corp.local)\n"
        f"{proto}  10.0.0.1  {port}  DC01  [-] corp.local\\admin:pw STATUS_LOGON_FAILURE\n"
    )
    adapter = FakeExegolAdapter(exec_router=lambda _cmd: ExecResult(stdout=raw, stderr="", exit_code=0))
    result = await exec_remote_command(
        adapter, "exegol-lab", proto.lower(), "10.0.0.1", "admin", "pw", "whoami", domain="corp.local"
    )
    assert result.exit_code == 0
    failure_lines = [line for line in result.stdout.splitlines() if "[-]" in line]
    assert failure_lines, result.stdout
    assert "corp.local\\admin:pw" in failure_lines[0]


async def test_the_remote_tool_returns_an_smb_login_failure_with_a_marker_in_the_password_as_exit_0_and_a_failure_line(
) -> None:
    # Tracer (G-03-4), end to end from the tool handler through the service to the parser fallback: a password
    # holding `[+]` must not cut the `[-]` marker off the failure line, and the backend's exit code 0 is passed
    # through unchanged (EXEC-08), the credential text unredacted (D-09).
    raw = (
        "SMB         10.0.0.1        445    DC01             [*] Windows Server 2019 Build 17763 x64 (name:DC01) "
        "(domain:corp.local)\n"
        "SMB         10.0.0.1        445    DC01             [-] corp.local\\admin:p[+]ss STATUS_LOGON_FAILURE\n"
    )
    adapter = FakeExegolAdapter(exec_router=lambda _cmd: ExecResult(stdout=raw, stderr="", exit_code=0))
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=adapter))
    result = await remote_tool.exec_remote_command(
        container_name="exegol-lab",
        protocol="smb",
        target_ip="10.0.0.1",
        username="admin",
        password="p[+]ss",
        command="whoami",
        domain="corp.local",
        ctx=ctx,  # type: ignore[arg-type]
    )
    assert result.exit_code == 0
    assert "[-] corp.local\\admin:p[+]ss STATUS_LOGON_FAILURE" in result.stdout


async def test_a_multi_target_remote_run_over_mcp_keeps_a_failure_line_printed_before_another_hosts_output() -> None:
    # Tracer (G-03-4, WR-04), over the real MCP call path: a CIDR target reaches the backend unchanged, DC01's `[-]`
    # line printed before WS01's `[+] Executed command` survives the parse in netexec's order, the backend's exit code
    # 0 is passed through (EXEC-08) and the credential text comes back unredacted (D-09).
    adapter = FakeExegolAdapter(exec_router=lambda _cmd: ExecResult(stdout=TWO_HOST_SMB_RUN, stderr="", exit_code=0))
    async with Client(build_server(None, adapter)) as client:
        result = await client.call_tool(
            "exec_remote_command",
            {
                "container_name": "exegol-lab",
                "protocol": "smb",
                "target_ip": "10.0.0.0/30",
                "username": "admin",
                "password": "pw",
                "command": "whoami",
                "domain": "corp",
            },
        )
    assert not result.is_error, result.content
    payload = result.structured_content
    assert payload is not None
    assert payload["exit_code"] == 0
    assert payload["stdout"] == (
        "SMB  10.0.0.1  445  DC01  [-] corp\\admin:pw STATUS_LOGON_FAILURE\n"
        "SMB  10.0.0.2  445  WS01  corp\\admin"
    )
    nxc_commands = [cmd for cmd in adapter.exec_commands if cmd.startswith("nxc ")]
    assert nxc_commands and nxc_commands[0].startswith("nxc smb 10.0.0.0/30 -u ")


@pytest.mark.parametrize(("raw", "expected"), SINGLE_TARGET_SUCCESS_RUNS)
async def test_a_single_target_remote_success_over_mcp_returns_only_the_command_output(raw: str, expected: str) -> None:
    # Tracer (WR-06) over the real MCP call path: the parsed stdout is only the command output (the fb938e7 result)
    # and the backend's exit code 0 is passed through (EXEC-08).
    adapter = FakeExegolAdapter(exec_router=lambda _cmd: ExecResult(stdout=raw, stderr="", exit_code=0))
    async with Client(build_server(None, adapter)) as client:
        result = await client.call_tool(
            "exec_remote_command",
            {
                "container_name": "exegol-lab",
                "protocol": "smb",
                "target_ip": "10.0.0.1",
                "username": "admin",
                "password": "pw",
                "command": "whoami",
                "domain": "corp",
            },
        )
    assert not result.is_error, result.content
    payload = result.structured_content
    assert payload is not None
    assert payload["exit_code"] == 0
    assert payload["stdout"] == expected


async def test_a_multi_target_ssh_run_over_mcp_names_the_host_that_rejected_the_login() -> None:
    # WR-07 over the real MCP call path: the agent can tell 10.0.0.1 rejected the login, the backend's exit code 0 is
    # passed through (EXEC-08) and the credential text comes back unredacted (D-09).
    adapter = FakeExegolAdapter(exec_router=lambda _cmd: ExecResult(stdout=TWO_HOST_SSH_RUN, stderr="", exit_code=0))
    async with Client(build_server(None, adapter)) as client:
        result = await client.call_tool(
            "exec_remote_command",
            {
                "container_name": "exegol-lab",
                "protocol": "ssh",
                "target_ip": "10.0.0.0/30",
                "username": "root",
                "password": "pw",
                "command": "id",
            },
        )
    assert not result.is_error, result.content
    payload = result.structured_content
    assert payload is not None
    assert payload["exit_code"] == 0
    assert payload["stdout"] == "SSH  10.0.0.1  22  10.0.0.1  [-] root:pw\nuid=0(root)"
    nxc_commands = [cmd for cmd in adapter.exec_commands if cmd.startswith("nxc ")]
    assert nxc_commands and nxc_commands[0].startswith("nxc ssh 10.0.0.0/30 -u ")


def test_the_remote_tool_description_states_the_backend_exit_code_truth() -> None:
    # G-03-3: the docstring is the LLM-facing tool description. It must say that netexec exits 0 on an
    # authentication failure and point to the `[-]` output line, keep its guidance blocks, and drop the old promise
    # of a nonzero code on failure. The service docstring states the same contract.
    tool_doc = inspect.getdoc(remote_tool.exec_remote_command) or ""
    service_doc = inspect.getdoc(exec_remote_command) or ""
    for doc in (tool_doc, service_doc):
        assert "exits 0" in doc
        assert "[-]" in doc
        assert "result with a non-zero exit code" not in doc
    for heading in ("IMPORTANT:", "RELATED TOOLS:", "SUPPORTED PROTOCOLS:", "Args:", "Returns:"):
        assert heading in tool_doc


def test_the_remote_descriptions_say_a_rejected_login_is_a_line_containing_the_marker() -> None:
    # G-03-4 (WR-03): the marker is not always at the front of the line. On SMB/WinRM/MSSQL/WMI/RDP (and SSH to a
    # host name) it follows the backend's protocol/host prefix, and get_job's raw output never starts with it. Both
    # docstrings must say "containing" (single or RST double backticks, a line wrap allowed) with the SMB example.
    tool_doc = inspect.getdoc(remote_tool.exec_remote_command) or ""
    service_doc = inspect.getdoc(exec_remote_command) or ""
    for doc in (tool_doc, service_doc):
        assert re.search(r"containing\s+`{1,2}\[-\]", doc), doc
        assert "starting with" not in doc
        for text in ("DC01", "STATUS_LOGON_FAILURE", "CORP\\user:password"):
            assert text in doc, text
    for kept in ("netexec is the current backend", "get_job", "IMPORTANT:", "RELATED TOOLS:", "SUPPORTED PROTOCOLS:",
                 "Args:", "Returns:"):
        assert kept in tool_doc, kept


async def test_remote_does_not_add_the_password_or_full_command_to_the_result() -> None:
    adapter = FakeExegolAdapter(exec_router=lambda _cmd: ExecResult(stdout="done", stderr="", exit_code=0))
    result = await exec_remote_command(
        adapter, "exegol-lab", "smb", "10.0.0.1", "admin", "SuperSecret123", "id", domain="CORP"
    )
    assert "SuperSecret123" not in result.stdout
    assert "SuperSecret123" not in result.stderr
    assert "nxc" not in result.stdout


async def test_remote_returns_credentials_the_command_prints_unredacted() -> None:
    # SEC-02: tool output is a finding for the operator, so a credential the remote command prints comes back as is.
    raw = (
        "SMB  10.0.0.1  445  10.0.0.1  [+] Executed command\n"
        "SMB  10.0.0.1  445  10.0.0.1  CORP\\admin:SuperSecret123\n"
    )
    adapter = FakeExegolAdapter(exec_router=lambda _cmd: ExecResult(stdout=raw, stderr="SuperSecret123", exit_code=0))
    result = await exec_remote_command(
        adapter, "exegol-lab", "smb", "10.0.0.1", "admin", "SuperSecret123", "type creds.txt", domain="CORP"
    )
    assert result.stdout == "CORP\\admin:SuperSecret123"
    assert result.stderr == "SuperSecret123"
