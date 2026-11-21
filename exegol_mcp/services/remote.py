r"""Remote-exec service (CORE-04): netexec command builder + output parser (pure, unit-testable) + the service.

SDK-free: it only talks to an ExegolPort. The netexec invocation is run through the shared bounded/background exec
path (services.execution.exec_command), so it inherits the same per-call timeout, on_timeout demotion, background
flag, output cap and kill-on-cancel as every other container command (D-17) — netexec is not special-cased. v1 bugs
fixed while porting: the never-matching case-sensitive prefix match and the whitespace-collapsing re-join.

SEC-01 hardening (Phase 3): every interpolated field of the netexec command (target, the assembled -u value,
password and command) is shlex-quoted on the single shell string the SDK evaluates exactly once via `eval "$CMD"`
(adapter/exec.py). Quoting the assembled -u value both blocks injection and fixes the v1 `DOMAIN\user` defect
(the backslash is preserved through the eval instead of being consumed as an escape). The backend specifics sit
behind one internal RemoteBackend seam (D-03); execution.py stays backend-agnostic.
"""

import shlex
from collections.abc import Callable
from dataclasses import dataclass

from exegol_mcp.adapter.port import ExegolPort
from exegol_mcp.models.config import ExecutionLimits
from exegol_mcp.models.exec import BoundedExecResult
from exegol_mcp.services import execution as execution_service
from exegol_mcp.services.containers import ensure_running
from exegol_mcp.services.jobs import JobStore

# Protocols netexec (nxc) supports, lowercase.
VALID_PROTOCOLS = ["ssh", "winrm", "smb", "mssql", "wmi", "rdp"]


def _build_netexec_command(
    protocol: str,
    target_ip: str,
    username: str,
    password: str,
    command: str,
    domain: str | None = None,
    port: int | None = None,
    use_powershell: bool = False,
) -> str:
    """Build a netexec (nxc) command string from the connection parameters.

    Args:
        protocol: Protocol to use (ssh, winrm, smb, mssql, wmi, rdp). The caller restricts it to VALID_PROTOCOLS,
            so it is the one field not quoted here.
        target_ip: Target IP address or hostname (shlex-quoted into the command).
        username: Username for authentication (shlex-quoted into the command).
        password: Password for authentication (shlex-quoted into the command).
        command: Command to execute on the target (shlex-quoted into the command).
        domain: Optional domain name; when set the user is sent as domain\\username (shlex-quoted as one field,
            which preserves the backslash through the single eval).
        port: Optional custom port (already range-checked by the caller), passed to netexec as its own
            ``--port`` option right after the target. netexec does not read a port from the target string: an
            IP:PORT target matches nothing and netexec silently does nothing (G-03-2).
        use_powershell: Use -X instead of -x so the command runs through PowerShell.

    Returns:
        The complete nxc command string, evaluated exactly once by the SDK's `eval "$CMD"` boundary (SEC-01).
    """
    # -x for a normal command, -X to run it through PowerShell.
    exec_flag = "-X" if use_powershell else "-x"

    # domain\username or plain username, assembled before quoting so the backslash is inside one quoted word.
    user_value = f"{domain}\\{username}" if domain else username

    # SEC-01: shlex-quote EVERY interpolated field on the single-eval shell string. The only unquoted values are
    # protocol (from the allowlist) and port (a range-checked int); target, the -u value, password and command
    # are untrusted text.
    cmd_parts = ["nxc", protocol, shlex.quote(target_ip)]

    # The port goes through netexec's own --port option, never suffixed onto the target. `is not None` rather
    # than truthiness, so the caller's range check is the only gate on the value.
    if port is not None:
        cmd_parts += ["--port", str(port)]

    cmd_parts += [
        "-u",
        shlex.quote(user_value),
        "-p",
        shlex.quote(password),
        exec_flag,
        shlex.quote(command),
    ]

    return " ".join(cmd_parts)


def _remove_netexec_prefix(line: str) -> str:
    """Strip the "PROTOCOL  IP  PORT  IP" log prefix netexec puts in front of each line.

    Fixes two v1 bugs: the prefix was matched with parts[0].upper() against a lowercase protocol list, so it
    never matched; and the remainder was rebuilt with ' '.join of the split tokens, collapsing every run of
    internal whitespace. Here the protocol is compared case-insensitively and the remainder is sliced off the
    original line (split with maxsplit=4), so the command output keeps its original spacing.

    Args:
        line: One line of netexec output, with or without the log prefix.

    Returns:
        The line with the prefix removed, or the original line when it carries no recognisable prefix.
    """
    if not line.strip():
        return ""

    parts = line.split()

    # A prefixed line starts with a protocol name followed by IP, PORT, IP (compare the protocol lowercased).
    if len(parts) >= 4 and parts[0].lower() in VALID_PROTOCOLS:

        def looks_like_ip(part: str) -> bool:
            # Drop a trailing :PORT before checking the dotted-quad shape.
            ip_part = part.split(":")[0]
            octets = ip_part.split(".")
            return len(octets) == 4 and all(octet.isdigit() and 0 <= int(octet) <= 255 for octet in octets)

        if looks_like_ip(parts[1]) and looks_like_ip(parts[3]):
            # Slice the original line after the fourth field so the remaining output keeps its spacing,
            # rather than ' '.join(parts[4:]) which collapses runs of whitespace.
            fields = line.strip().split(None, 4)
            return fields[4] if len(fields) == 5 else ""

    # No recognisable prefix: leave the line untouched.
    return line


def _parse_netexec_output(output: str) -> str:
    """Extract the command output from raw netexec output, dropping its log lines.

    netexec logs each line with a "PROTOCOL  HOST  PORT  NAME  <message>" prefix, where NAME is the IP on SSH and
    the NetBIOS name on SMB-family protocols, and marks the start of the command output with "[+] Executed
    command". The lines after that marker are kept with their prefix stripped, and a best-effort fallback runs when
    the main loop collects nothing.

    A line containing "[-]" is kept wherever it appears, in netexec's order, unless the same host, identified by
    column 2 of the prefix (the address netexec connected to, never the target-controlled NAME), prints its first
    marker after it: that host authenticated and ran the command, so such a line is a diagnostic, not a rejected
    login. "[-]" is checked before the marker match, so a failure line whose credential holds the marker text is
    never read as the marker. When the output names more than one host, a kept "[-]" line keeps its whole netexec
    line, prefix included, so the agent can tell which host rejected the login; with one host, only the prefix is
    stripped, as before.

    The fallback keeps a line containing "[-]" whole in the same way (prefix included when more than one host
    appears, otherwise only the prefix is stripped), so a failure line is never cut at a "[+]"/"[*]" inside its
    credential text.

    Args:
        output: Raw stdout from the netexec invocation.

    Returns:
        The cleaned command output, or the original output when nothing could be parsed.
    """
    if not output:
        return ""

    lines = output.split("\n")
    cleaned_lines: list[str] = []
    failures_before_marker: list[str] = []
    found_executed = False

    def host_of(line: str) -> str | None:
        # Column 2 of the netexec prefix is the address netexec connected to, which the target never controls,
        # unlike the column-4 name. Same shape test as _remove_netexec_prefix, without its IP check: column 2 can
        # be a hostname given as the target.
        parts = line.split()
        if len(parts) >= 4 and parts[0].lower() in VALID_PROTOCOLS:
            return parts[1]
        return None

    hosts = [host_of(line) for line in lines]

    # Maintainer decision 2 (WR-07): in a multi-target run the netexec prefix is a kept failure line's only host
    # identity (SSH puts the IP in column 4, so the strip removes it), so the whole line is kept. With one host in the
    # output the prefix is stripped exactly as before, which keeps single-target output byte-identical. The count is
    # read from the output: a sweep where only one host prints anything parses like a single-target run (T-03-26).
    multi_target = len({host for host in hosts if host is not None}) > 1

    def failure_text(line: str) -> str:
        return line if multi_target else _remove_netexec_prefix(line)

    # Each host's first real marker line: one the main loop treats as the marker, so it holds no `[-]`.
    first_marker: dict[str, int] = {}
    for index, line in enumerate(lines):
        host = hosts[index]
        if host is not None and "[-]" not in line and "[+] Executed command" in line:
            first_marker.setdefault(host, index)

    for index, line in enumerate(lines):
        # netexec exits 0 on an authentication failure, so a `[-]` line is the agent's only failure signal. In a
        # multi-target run another host's "[+] Executed command" can follow it (G-03-4, WR-04), so keep it whole
        # wherever it sits. Checking it before the marker match also stops a failure line whose credential holds
        # the marker text from being consumed as the marker.
        if "[-]" in line:
            # Maintainer decision 1 (WR-06): a host whose first real marker follows this line authenticated and ran
            # the command, so the line is a diagnostic (an exec-method fallback, or `[-]` in a password or in a
            # target-controlled banner), not a rejected login. Keyed on column 2, so a host cannot hide another
            # host's rejection by reusing its name. A host that never runs the command keeps every `[-]` line.
            host = hosts[index]
            if host is not None and first_marker.get(host, -1) > index:
                continue
            cleaned_failure = failure_text(line)
            if cleaned_failure.strip():
                if found_executed:
                    cleaned_lines.append(cleaned_failure)
                else:
                    failures_before_marker.append(cleaned_failure)
            continue

        # The "[+] Executed command" marker precedes the real output.
        if "[+] Executed command" in line:
            found_executed = True
            continue

        if found_executed:
            cleaned_line = _remove_netexec_prefix(line)
            if cleaned_line:
                cleaned_lines.append(cleaned_line)

    # Every line before the first marker precedes every line after it, so prepending keeps netexec's order. Only
    # when the loop collected output: an empty result defers to the fallback below, which keeps `[-]` lines itself,
    # so a single-target failed login parses exactly as before.
    if cleaned_lines:
        cleaned_lines = failures_before_marker + cleaned_lines

    # Without the marker, try to recover output lines directly.
    if not cleaned_lines:
        for line in lines:
            # The backend exits 0 on an authentication failure, so a `[-]` line is the agent's only failure signal
            # (G-03-4). Keep it whole: a `[+]` or `[*]` inside a credential or the error text must not cut it away.
            if "[-]" in line:
                cleaned = failure_text(line)
                if cleaned.strip():
                    cleaned_lines.append(cleaned)
                continue

            if "[*]" in line or "[+]" in line:
                if "[+] Executed command" not in line:
                    marker_pos = max(line.find("[*]"), line.find("[+]"))
                    if marker_pos != -1:
                        after_marker = line[marker_pos + 3 :].strip()
                        cleaned = _remove_netexec_prefix(after_marker)
                        if cleaned and cleaned not in ["Executed command", "Shell access!"]:
                            cleaned_lines.append(cleaned)
                continue

            cleaned = _remove_netexec_prefix(line)
            if cleaned and cleaned.strip():
                cleaned_lines.append(cleaned)

    result = "\n".join(cleaned_lines).strip()
    return result if result else output


@dataclass(frozen=True)
class RemoteBackend:
    """One remote-execution backend seam (D-03): it owns the backend-specific knowledge of how to turn the typed
    connection parameters into an invocation string and how to parse that invocation's raw output.

    netexec is the sole backend today (NETEXEC_BACKEND). The seam exists so a future second backend plugs in here
    without changing the agent-facing exec_remote_command contract (D-04) and without any netexec special-casing
    leaking into services.execution, which stays backend-agnostic (D-03/D-17). The seam is confined to this module.
    """

    name: str
    build_command: Callable[..., str]
    parse_output: Callable[[str], str]


# The current (and only) remote-execution backend. exec_remote_command talks to this seam, never to the module
# functions directly, so the netexec specifics sit behind one named object.
NETEXEC_BACKEND = RemoteBackend(
    name="netexec",
    build_command=_build_netexec_command,
    parse_output=_parse_netexec_output,
)


async def exec_remote_command(
    adapter: ExegolPort,
    container_name: str,
    protocol: str,
    target_ip: str,
    username: str,
    password: str,
    command: str,
    domain: str | None = None,
    port: int | None = None,
    use_powershell: bool = False,
    *,
    timeout: float | None = None,
    on_timeout: str = "background",
    background: bool = False,
    limits: ExecutionLimits | None = None,
    jobs: JobStore | None = None,
    tag: str | None = None,
    auto_start: bool = False,
    start_timeout_s: float | None = None,
) -> BoundedExecResult:
    """Run a command on a remote host via netexec, bounded exactly like exec_command (D-17).

    The built netexec command runs through the shared bounded/background exec path (services.execution.exec_command),
    so a long sweep or spray inherits the same treatment as any other container command: a per-call ``timeout``
    (operator default when omitted, clamped to the operator maximum), an ``on_timeout`` branch (default
    ``background``, which demotes an overrun to a pollable job carrying the partial output; ``kill``, which
    terminates it and returns the partial output with exit_code=None), a ``background`` flag that returns a job id
    at once, tail-capped output (EXEC-03) and cancellation that kills the netexec process group (EXEC-07).

    A connection or authentication failure is returned as a result, not an exception (the v1 convention). The exit
    code is the backend's own, passed through unchanged: netexec exits 0 when authentication fails, and the failure
    shows in the output as a line containing ``[-]``, which can follow netexec's protocol/host prefix (for example
    ``SMB 10.0.0.1 445 DC01 [-] CORP\\user:password STATUS_LOGON_FAILURE``), in the parsed result and the raw
    get_job output alike; the fallback parse never cuts a line containing ``[-]``. No exit code is synthesized from
    the parsed output, so results stay faithful (EXEC-08) and this service stays backend-generic. The password and
    the built command are never logged or reported as progress on any path. The result carries netexec's output
    unredacted (SEC-02), so it can contain credentials netexec prints; the output cap / truncation marker reports
    volume dropped, it is not redaction.

    Parsing: a synchronous foreground result keeps netexec's log-line parse on stdout. A background or timed-out
    remote command is a job whose RAW netexec output is read via get_job — the parse cannot run incrementally on
    those reads, so the job returns the raw netexec output (stated in the tool docstring).

    Args:
        adapter: The Exegol port.
        container_name: Name of the container netexec runs in.
        protocol: Protocol to use (ssh, winrm, smb, mssql, wmi, rdp), case-insensitive.
        target_ip: Target IP address or hostname.
        username: Username for authentication.
        password: Password for authentication.
        command: Command to run on the target.
        domain: Optional domain for Windows protocols.
        port: Optional port of the remote service, 1-65535, passed to the backend as its own option (netexec's
            ``--port``). None uses the protocol's default port.
        use_powershell: Use the PowerShell flag (-X) instead of -x.
        timeout: Per-call foreground timeout in seconds; None applies the operator default, any value is clamped
            to the operator maximum. Must be positive when given.
        on_timeout: What to do when the foreground timeout expires: "background" (default, demote to a job) or
            "kill" (terminate and return the partial output).
        background: Start the netexec command in the background and return a job id immediately (EXEC-04).
        limits: Execution limits (timeout default/max, output cap, concurrency cap, kill grace). None uses defaults.
        jobs: The background-job store. Required for background; the foreground timeout/promotion path also uses it
            (a foreground call without a store runs inline and unbounded, for ad-hoc service use).
        tag: The env tag/job id to launch the foreground command under; the tool mints it up front so a client
            cancellation can kill the netexec process group by tag (EXEC-07). A fresh opaque id is minted when omitted.
        auto_start: Start the container first when it is stopped (LIFE-07, D-15). The check and the start happen
            after the protocol/port validation and before the backend command is built.
        start_timeout_s: Container start timeout in seconds for auto_start; None uses the built-in
            container_op_timeout_s default (D-18). The tool passes the operator-configured value.

    Returns:
        A synchronous foreground result: a BoundedExecResult with the parsed netexec output, the invocation's
        stderr, the real exit code, timed_out False and job_id None. A background or timed-out remote command:
        the BoundedExecResult from the bounded path (raw netexec output, timed_out/job_id set as appropriate).

    Raises:
        ValueError: If the protocol is not one of VALID_PROTOCOLS, the port is outside 1-65535, the container name
            is unknown, the timeout is not positive, or on_timeout is not "kill"/"background".
        RuntimeError: If the container is not running and auto_start is false, an auto_start does not finish
            within start_timeout_s, or background is requested without a job store.
    """
    # 1. Validate the protocol and the port before touching the container (both are the agent's to fix). A port
    #    outside the TCP range would otherwise reach the backend and fail silently or obscurely.
    protocol_normalized = protocol.lower()
    if protocol_normalized not in VALID_PROTOCOLS:
        raise ValueError(f"Invalid protocol '{protocol}'. Valid protocols: {', '.join(VALID_PROTOCOLS)}")
    if port is not None and not 1 <= port <= 65535:
        raise ValueError(f"Invalid port {port}: must be between 1 and 65535 (omit port to use the protocol's default)")

    # 2. The container must exist and run, checked (and, on an explicit auto_start, started under its own D-18
    #    bound) before the password-bearing command is ever built (D-15).
    await ensure_running(adapter, container_name, auto_start=auto_start, timeout_s=start_timeout_s)

    # 3. Build the invocation through the backend seam (every interpolated field shlex-quoted, SEC-01; never
    #    logged or reported as progress). The seam owns the backend-specific build; this function stays generic.
    backend_cmd = NETEXEC_BACKEND.build_command(
        protocol=protocol_normalized,
        target_ip=target_ip,
        username=username,
        password=password,
        command=command,
        domain=domain,
        port=port,
        use_powershell=use_powershell,
    )

    # 4. Run it through the shared bounded/background exec path (D-17): same timeout clamp, on_timeout demotion,
    #    background flag, output cap and kill-on-cancel as exec_command. No special-casing of netexec (project rule).
    result = await execution_service.exec_command(
        adapter,
        container_name,
        backend_cmd,
        timeout=timeout,
        on_timeout=on_timeout,
        limits=limits,
        background=background,
        jobs=jobs,
        tag=tag,
        # The container runs at this point, so the shared path never attempts a second start.
        auto_start=False,
    )

    # 5. A background or timed-out remote command is a job: its raw netexec output is read via get_job (the parse
    #    cannot run incrementally on those reads), so return it raw. A synchronous foreground result keeps the parse.
    if background or result.timed_out or result.job_id is not None:
        return result
    return result.model_copy(update={"stdout": NETEXEC_BACKEND.parse_output(result.stdout)})
