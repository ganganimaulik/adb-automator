# macOS Sleep Prevention Rule

When executing long-running tasks or background processes on macOS, always consider using `caffeinate` to prevent system sleep during execution.

## Automatic sleep prevention in adbagent
`adbagent` CLI entrypoint in `adbagent/cli.py` uses `prevent_sleep()` (`caffeinate -w <pid> -d -i -s`) to automatically keep macOS system and display screen awake while any `adbagent` command is running.

## Command execution guidelines
For any long-running shell background tasks or continuous loops run via `run_command` on macOS:
- Prefix long-running background tasks with `caffeinate -d -i` if display screen and system idle sleep should be prevented.
