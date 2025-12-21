import shlex
from typing import Optional

from mcp.server.fastmcp import Context

from exegol_mcp.utils.exegol_utils import get_container_by_name, check_exegol_readiness
from exegol_mcp.mcp_app import mcp_server
from exegol_mcp.models.container import ExecutionResult

# Valid protocols supported by netexec
VALID_PROTOCOLS = ["ssh", "winrm", "smb", "mssql", "wmi", "rdp"]


def _build_netexec_command(
    protocol: str,
    target_ip: str,
    username: str,
    password: str,
    command: str,
    domain: Optional[str] = None,
    port: Optional[int] = None,
    use_powershell: bool = False
) -> str:
    """
    Build a netexec command based on protocol and parameters.
    
    Args:
        protocol: Protocol to use (ssh, winrm, smb, mssql, wmi, rdp)
        target_ip: Target IP address
        username: Username for authentication
        password: Password for authentication
        command: Command to execute
        domain: Optional domain name
        port: Optional custom port
        use_powershell: Use -X flag instead of -x (for PowerShell commands)
    
    Returns:
        Complete netexec command string
    """
    # Determine the flag to use (-x for normal commands, -X for PowerShell)
    exec_flag = "-X" if use_powershell else "-x"
    
    # Build command parts
    # Add port to IP if specified (format: IP:PORT)
    target = f"{target_ip}:{port}" if port else target_ip
    cmd_parts = ["nxc", protocol, target]
    
    # Add username
    if domain:
        # Format: domain\username or username@domain
        cmd_parts.extend(["-u", f"{domain}\\{username}"])
    else:
        cmd_parts.extend(["-u", username])
    
    # Add password (properly escaped with shlex.quote)
    cmd_parts.extend(["-p", shlex.quote(password)])
    
    # Add execution flag and command (command needs to be quoted as it may contain spaces)
    cmd_parts.extend([exec_flag, shlex.quote(command)])
    
    return " ".join(cmd_parts)


async def _execute_remote_command(
    container_name: str,
    protocol: str,
    target_ip: str,
    username: str,
    password: str,
    command: str,
    domain: Optional[str] = None,
    port: Optional[int] = None,
    use_powershell: bool = False,
    ctx: Optional[Context] = None
) -> ExecutionResult:
    """
    Execute a command remotely through various protocols.
    
    Args:
        container_name: Name of the Exegol container
        protocol: Protocol to use
        target_ip: Target IP address
        username: Username for authentication
        password: Password for authentication
        command: Command to execute
        domain: Optional domain name
        port: Optional custom port
        use_powershell: Use PowerShell flag (-X) instead of normal flag (-x)
        ctx: Optional MCP context for logging
    
    Returns:
        ExecutionResult with command output
    """
    # Check if container exists and is running
    container = get_container_by_name(container_name)
    if not container:
        error_msg = f"Container '{container_name}' not found"
        if ctx:
            await ctx.error(error_msg)
        raise ValueError(error_msg)
    
    if not container.isRunning():
        error_msg = f"Container '{container_name}' is not running"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Build netexec command
    netexec_cmd = _build_netexec_command(
        protocol=protocol,
        target_ip=target_ip,
        username=username,
        password=password,
        command=command,
        domain=domain,
        port=port,
        use_powershell=use_powershell
    )
    
    if ctx:
        await ctx.info(f"Executing command via {protocol}: {command}")
        await ctx.info(f"Full netexec command: {netexec_cmd}")
    
    # Execute command in container
    exit_code, output = await container.exec_raw(netexec_cmd)
    
    # Parse netexec output to extract only the command output
    # netexec outputs logs in format: "PROTOCOL  IP  PORT  IP  [*] or [+] message"
    # The actual command output comes after "[+] Executed command"
    cleaned_output = _parse_netexec_output(output)
    
    if ctx:
        if exit_code == 0:
            await ctx.info(f"Command executed successfully")
        else:
            await ctx.warning(f"Command execution failed with exit code {exit_code}")
    
    return ExecutionResult(exit_code=exit_code, output=cleaned_output)


def _parse_netexec_output(output: str) -> str:
    """
    Parse netexec output to extract only the command output.
    Removes netexec debug/log messages and returns only the actual command output.
    
    Netexec output format:
    PROTOCOL  IP  PORT  IP  [*] or [+] message
    PROTOCOL  IP  PORT  IP  actual command output
    
    Args:
        output: Raw output from netexec command
    
    Returns:
        Cleaned output containing only the command result
    """
    if not output:
        return ""
    
    lines = output.split('\n')
    cleaned_lines = []
    found_executed = False
    
    for line in lines:
        # Look for the "[+] Executed command" marker
        if "[+] Executed command" in line:
            found_executed = True
            continue
        
        # After finding "Executed command", extract the actual output
        if found_executed:
            # Remove netexec prefix: "PROTOCOL  IP  PORT  IP  "
            # Pattern: protocol name (uppercase), IP, port number, IP again, then the actual output
            cleaned_line = _remove_netexec_prefix(line)
            if cleaned_line:  # Only add non-empty lines
                cleaned_lines.append(cleaned_line)
    
    # If we didn't find the "Executed command" marker, try to extract output differently
    if not cleaned_lines:
        # Look for lines after authentication that contain actual output
        # Skip log lines with [*] or [+] markers (except when they contain the output)
        for line in lines:
            # Skip pure log lines
            if "[*]" in line or "[+]" in line:
                # Check if this line contains actual output after the marker
                # Some netexec versions output results inline with markers
                if "[+] Executed command" not in line:
                    # Try to extract content after the marker
                    marker_pos = max(line.find("[*]"), line.find("[+]"))
                    if marker_pos != -1:
                        after_marker = line[marker_pos + 3:].strip()
                        # Remove netexec prefix from what's after the marker
                        cleaned = _remove_netexec_prefix(after_marker)
                        if cleaned and cleaned not in ["Executed command", "Shell access!"]:
                            cleaned_lines.append(cleaned)
                continue
            
            # Try to clean lines that might be output
            cleaned = _remove_netexec_prefix(line)
            if cleaned and cleaned.strip():
                cleaned_lines.append(cleaned)
    
    result = '\n'.join(cleaned_lines).strip()
    return result if result else output  # Return original if we couldn't parse


def _remove_netexec_prefix(line: str) -> str:
    """
    Remove netexec prefix from a line.
    Netexec prefixes lines with: "PROTOCOL  IP  PORT  IP  "
    
    Args:
        line: Line that may contain netexec prefix
    
    Returns:
        Line with prefix removed, or original line if no prefix found
    """
    if not line.strip():
        return ""
    
    # Pattern: protocol name (uppercase, 2-6 chars), spaces, IP, spaces, port, spaces, IP, spaces
    # Try to match and remove this pattern
    parts = line.split()
    
    # Check if line starts with a protocol name
    if len(parts) >= 4 and parts[0].upper() in VALID_PROTOCOLS:
        # Check if parts[1] and parts[3] look like IP addresses
        def looks_like_ip(part: str) -> bool:
            # Remove port if present (IP:PORT)
            ip_part = part.split(':')[0]
            octets = ip_part.split('.')
            return len(octets) == 4 and all(octet.isdigit() and 0 <= int(octet) <= 255 for octet in octets)
        
        if looks_like_ip(parts[1]) and looks_like_ip(parts[3]):
            # This looks like a netexec log line, extract everything after the 4th part
            # Join parts starting from index 4 (after PROTOCOL IP PORT IP)
            return ' '.join(parts[4:])
    
    # If no prefix pattern found, return original line
    return line


@mcp_server.tool()
async def execute_remote_command(
    container_name: str,
    protocol: str,
    target_ip: str,
    username: str,
    password: str,
    command: str,
    domain: Optional[str] = None,
    port: Optional[int] = None,
    use_powershell: bool = False,
    ctx: Context = None
) -> ExecutionResult:
    """
    Execute a command remotely through various protocols (SSH, WinRM, SMB, MSSQL, WMI, RDP).
    This tool directly executes the command without creating a persistent session.
    
    Args:
        container_name: Name of the Exegol container to use
        protocol: Protocol to use (ssh, winrm, smb, mssql, wmi, rdp)
        target_ip: Target IP address
        username: Username for authentication
        password: Password for authentication
        command: Command to execute
        domain: Optional domain name (for Windows protocols like smb, winrm)
        port: Optional custom port
        use_powershell: Use PowerShell flag (-X) instead of normal flag (-x). 
                       Useful for SMB PowerShell commands or WinRM commands.
    Returns:
        ExecutionResult with command output and exit code
    """
    await ctx.info(f"Executing command via {protocol} to {target_ip} as {username}")
    await check_exegol_readiness(ctx)
    
    # Validate protocol
    protocol_normalized = protocol.lower()
    if protocol_normalized not in VALID_PROTOCOLS:
        error_msg = f"Invalid protocol '{protocol}'. Valid protocols: {', '.join(VALID_PROTOCOLS)}"
        await ctx.error(error_msg)
        raise ValueError(error_msg)
    
    # Execute command directly
    return await _execute_remote_command(
        container_name=container_name,
        protocol=protocol_normalized,
        target_ip=target_ip,
        username=username,
        password=password,
        command=command,
        domain=domain,
        port=port,
        use_powershell=use_powershell,
        ctx=ctx
    )

