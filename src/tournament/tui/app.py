"""Tournament TUI - Terminal dashboard for miners."""

import argparse
import sys
import time

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from tournament.tui.client import (
    MockClient,
    SubmissionDetail,
    TournamentClient,
    TournamentData,
    format_time_ago,
)

console = Console()


def create_chart(history: list[dict], width: int = 50, height: int = 8) -> Text:
    """Create an ASCII chart of TPS over time."""
    if not history:
        return Text("No data available", style="dim italic", justify="center")

    # Sort by timestamp and get TPS values
    sorted_history = sorted(history, key=lambda x: x.get("timestamp", ""))
    tps_values = [h.get("tps", 0) for h in sorted_history]

    if not tps_values:
        return Text("No data available", style="dim italic", justify="center")

    min_tps = min(tps_values)
    max_tps = max(tps_values)
    tps_range = max_tps - min_tps if max_tps != min_tps else 1

    # Unicode block characters for different heights
    blocks = " ▁▂▃▄▅▆▇█"

    # Build the chart
    lines = []

    # Y-axis labels and chart area
    for row in range(height, 0, -1):
        threshold = min_tps + (tps_range * row / height)
        if row == height:
            label = f"{int(max_tps):>5} │"
        elif row == 1:
            label = f"{int(min_tps):>5} │"
        elif row == height // 2:
            mid_val = int(min_tps + tps_range / 2)
            label = f"{mid_val:>5} │"
        else:
            label = "      │"

        # Build the row
        row_chars = []
        for i, tps in enumerate(tps_values):
            normalized = (tps - min_tps) / tps_range
            bar_height = normalized * height

            if bar_height >= row:
                row_chars.append("█")
            elif bar_height >= row - 1:
                # Partial block
                partial = bar_height - (row - 1)
                block_idx = int(partial * 8)
                row_chars.append(blocks[min(block_idx, 8)])
            else:
                row_chars.append(" ")

        # Pad or truncate to fit width
        chart_width = width - 7  # Account for y-axis label
        if len(row_chars) > chart_width:
            # Sample evenly
            step = len(row_chars) / chart_width
            row_chars = [row_chars[int(i * step)] for i in range(chart_width)]
        else:
            # Stretch to fill
            if row_chars:
                stretched = []
                step = chart_width / len(row_chars)
                for i in range(chart_width):
                    stretched.append(row_chars[min(int(i / step), len(row_chars) - 1)])
                row_chars = stretched

        lines.append(label + "".join(row_chars))

    # X-axis
    x_axis = "      └" + "─" * (width - 7)
    lines.append(x_axis)

    # Time labels
    if sorted_history:
        first_time = sorted_history[0].get("timestamp", "")[:10]
        last_time = sorted_history[-1].get("timestamp", "")[:10]
        time_label = f"       {first_time}" + " " * (width - 18 - len(first_time)) + last_time
        lines.append(time_label)

    chart_text = Text()
    for i, line in enumerate(lines):
        if i < len(lines) - 2:  # Chart bars
            chart_text.append(line[:7])  # Y-axis label
            chart_text.append(line[7:], style="green")
        else:
            chart_text.append(line, style="dim")
        if i < len(lines) - 1:
            chart_text.append("\n")

    return chart_text


def create_chart_panel(data: TournamentData) -> Panel:
    """Create the TPS history chart panel."""
    # Use full width - get console width dynamically
    chart = create_chart(data.history, width=90, height=6)
    return Panel(
        chart,
        title="[bold]TPS History[/bold]",
        border_style="yellow",
    )


def create_stats_panel(data: TournamentData) -> Panel:
    """Create the stats overview panel."""
    overview = data.overview

    stats = Table.grid(padding=(0, 4))
    stats.add_column(justify="center")
    stats.add_column(justify="center")
    stats.add_column(justify="center")
    stats.add_column(justify="center")

    stats.add_row(
        Text(f"24h Submissions\n{overview.get('submissions_24h', 0)}", justify="center"),
        Text(f"Top TPS\n{overview.get('current_top_score', 0):.1f}", justify="center"),
        Text(f"Active Miners\n{overview.get('active_miners', 0)}", justify="center"),
        Text(f"Total\n{overview.get('total_submissions', 0)}", justify="center"),
    )

    return Panel(Align.center(stats), title="[bold]Stats[/bold]", border_style="blue")


def create_validator_panel(data: TournamentData) -> Panel:
    """Create the validator status panel."""
    validator = data.validator
    queue = data.queue

    status = validator.get("status", "unknown")
    status_color = "green" if status == "running" else "red"
    status_dot = f"[{status_color}]●[/{status_color}]"

    current_eval = validator.get("current_evaluation")
    eval_text = "None"
    if current_eval and current_eval.get("miner_uid") is not None:
        eval_text = f"UID {current_eval['miner_uid']}"

    uptime = validator.get("uptime", "N/A")
    evals_1h = validator.get("evaluations_completed_1h", 0)

    pending = queue.get("pending_count", 0)
    running = queue.get("running_count", 0)

    info = (
        f"{status_dot} {status.upper()}  |  "
        f"Evaluating: {eval_text}  |  "
        f"Evals/hr: {evals_1h}  |  "
        f"Queue: {pending} pending, {running} running  |  "
        f"Uptime: {uptime}"
    )

    return Panel(
        Text.from_markup(info, justify="center"),
        title="[bold]Validator[/bold]",
        border_style="cyan",
    )


def create_leaderboard_table(
    data: TournamentData, selected_idx: int | None = None, active: bool = True
) -> Panel:
    """Create the leaderboard table with selection highlight."""
    table = Table(expand=True, box=None, show_header=True, header_style="bold")

    table.add_column("#", justify="right", width=3)
    table.add_column("Rank", justify="right", width=5)
    table.add_column("UID", justify="right", width=6)
    table.add_column("TPS", justify="right", style="green", width=10)
    table.add_column("Evals", justify="right", width=6)
    table.add_column("Submitted", justify="right", width=10)

    for idx, entry in enumerate(data.leaderboard[:10]):
        rank = entry.get("rank", "?")
        rank_style = ""
        if rank == 1:
            rank_style = "[yellow]"
        elif rank == 2:
            rank_style = "[white]"
        elif rank == 3:
            rank_style = "[#cd7f32]"

        rank_display = f"{rank_style}{rank}[/]" if rank_style else str(rank)

        # Highlight selected row
        is_selected = active and selected_idx == idx
        row_style = "reverse" if is_selected else ""

        table.add_row(
            f"[cyan]{idx + 1}[/]",
            rank_display,
            str(entry.get("miner_uid", "?")),
            f"{entry.get('final_score', 0):.2f}",
            str(entry.get("num_evaluations", 0)),
            format_time_ago(entry.get("created_at")),
            style=row_style,
        )

    if not data.leaderboard:
        table.add_row("-", "-", "-", "-", "-", "-")

    border = "green" if active else "dim"
    return Panel(table, title="[bold]Leaderboard[/bold]", border_style=border)


def create_recent_table(
    data: TournamentData, selected_idx: int | None = None, active: bool = False
) -> Panel:
    """Create the recent submissions table with selection highlight."""
    table = Table(expand=True, box=None, show_header=True, header_style="bold")

    table.add_column("#", justify="right", width=3)
    table.add_column("UID", justify="right", width=6)
    table.add_column("Status", justify="left", width=12)
    table.add_column("TPS", justify="right", width=10)
    table.add_column("Submitted", justify="right", width=10)

    status_colors = {
        "pending": "yellow",
        "validating": "yellow",
        "evaluating": "cyan",
        "finished": "green",
        "failed_validation": "red",
        "error": "red",
    }

    for idx, entry in enumerate(data.recent[:10]):
        status = entry.get("status", "unknown")
        color = status_colors.get(status, "white")

        score = entry.get("final_score")
        score_display = f"{score:.2f}" if score and score > 0 else "-"

        is_selected = active and selected_idx == idx
        row_style = "reverse" if is_selected else ""

        table.add_row(
            f"[cyan]{idx + 1}[/]",
            str(entry.get("miner_uid", "?")),
            f"[{color}]{status}[/{color}]",
            score_display,
            format_time_ago(entry.get("created_at")),
            style=row_style,
        )

    if not data.recent:
        table.add_row("-", "-", "-", "-", "-")

    border = "magenta" if active else "dim"
    return Panel(table, title="[bold]Recent Activity[/bold]", border_style=border)


def create_dashboard_layout(
    data: TournamentData,
    leaderboard_idx: int | None = None,
    recent_idx: int | None = None,
    active_panel: str = "leaderboard",
    demo: bool = False,
) -> Layout:
    """Create the full dashboard layout."""
    layout = Layout()

    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="stats", size=5),
        Layout(name="validator", size=3),
        Layout(name="chart", size=10),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )

    title = "τemplar crusades"
    if demo:
        title += "  [yellow][DEMO][/yellow]"

    layout["header"].update(
        Panel(
            Text.from_markup(title, justify="center", style="bold white"),
            border_style="white",
        )
    )

    layout["stats"].update(create_stats_panel(data))
    layout["validator"].update(create_validator_panel(data))
    layout["chart"].update(create_chart_panel(data))

    layout["main"].split_row(
        Layout(name="leaderboard", ratio=3),
        Layout(name="recent", ratio=2),
    )

    layout["leaderboard"].update(
        create_leaderboard_table(
            data,
            selected_idx=leaderboard_idx,
            active=(active_panel == "leaderboard"),
        )
    )
    layout["recent"].update(
        create_recent_table(
            data,
            selected_idx=recent_idx,
            active=(active_panel == "recent"),
        )
    )

    footer_text = (
        "[cyan]↑↓[/] Navigate  "
        "[cyan]←→[/] Switch panel  "
        "[cyan]Enter[/] View details  "
        "[cyan]r[/] Refresh  "
        "[cyan]q[/] Quit"
    )
    layout["footer"].update(
        Panel(Text.from_markup(footer_text, justify="center"), border_style="dim")
    )

    return layout


def create_submission_header(detail: SubmissionDetail) -> Panel:
    """Create the submission header panel."""
    sub = detail.submission

    status = sub.get("status", "unknown")
    status_colors = {
        "pending": "yellow",
        "validating": "yellow",
        "evaluating": "cyan",
        "finished": "green",
        "failed_validation": "red",
        "error": "red",
    }
    status_color = status_colors.get(status, "white")

    grid = Table.grid(padding=(0, 3))
    grid.add_column(justify="left")
    grid.add_column(justify="left")
    grid.add_column(justify="left")
    grid.add_column(justify="left")

    hotkey = sub.get("miner_hotkey", "N/A")
    if len(hotkey) > 16:
        hotkey = hotkey[:8] + "..." + hotkey[-8:]

    code_hash = sub.get("code_hash", "N/A")
    if len(code_hash) > 16:
        code_hash = code_hash[:16] + "..."

    grid.add_row(
        f"[bold]UID:[/] {sub.get('miner_uid', 'N/A')}",
        f"[bold]Status:[/] [{status_color}]{status}[/{status_color}]",
        f"[bold]TPS:[/] [green]{sub.get('final_score', 0):.2f}[/green]",
        f"[bold]Submitted:[/] {format_time_ago(sub.get('created_at'))}",
    )
    grid.add_row(
        f"[bold]Hotkey:[/] {hotkey}",
        f"[bold]Hash:[/] {code_hash}",
        "",
        "",
    )

    error = sub.get("error_message")
    if error:
        grid.add_row(f"[red][bold]Error:[/] {error}[/red]", "", "", "")

    return Panel(grid, title="[bold]Submission Details[/bold]", border_style="blue")


def create_evaluations_table(detail: SubmissionDetail) -> Panel:
    """Create the evaluations table."""
    table = Table(expand=True, box=None)
    table.add_column("#", justify="right", width=4)
    table.add_column("TPS", justify="right", style="green", width=12)
    table.add_column("Tokens", justify="right", width=12)
    table.add_column("Wall Time", justify="right", width=12)
    table.add_column("Status", justify="center", width=8)
    table.add_column("Time", justify="right", width=12)

    total_tps = 0.0
    count = 0

    for idx, eval_data in enumerate(detail.evaluations, 1):
        tps = eval_data.get("tokens_per_second", 0)
        total_tps += tps
        count += 1

        success = eval_data.get("success", False)
        status_display = "[green]✓[/green]" if success else "[red]✗[/red]"

        table.add_row(
            str(idx),
            f"{tps:.2f}",
            str(eval_data.get("total_tokens", 0)),
            f"{eval_data.get('wall_time_seconds', 0):.2f}s",
            status_display,
            format_time_ago(eval_data.get("created_at")),
        )

    if not detail.evaluations:
        table.add_row("-", "-", "-", "-", "-", "-")

    avg_tps = total_tps / count if count > 0 else 0
    title = f"[bold]Evaluations[/bold] (Avg TPS: [green]{avg_tps:.2f}[/green])"

    return Panel(table, title=title, border_style="cyan")


def create_code_panel(detail: SubmissionDetail, scroll_offset: int = 0) -> Panel:
    """Create the syntax-highlighted code panel."""
    code = detail.code

    if not code:
        content = Text("Code not available", style="dim italic", justify="center")
    else:
        content = Syntax(
            code,
            "python",
            theme="monokai",
            line_numbers=True,
            word_wrap=False,
            start_line=1,
        )

    code_hash = detail.submission.get("code_hash", "")[:16]
    title = "[bold]Code[/bold]"
    if code_hash:
        title += f" [dim](hash: {code_hash}...)[/dim]"

    return Panel(content, title=title, border_style="green")


def create_submission_layout(detail: SubmissionDetail, show_code: bool = True) -> Layout:
    """Create the submission detail layout."""
    layout = Layout()

    if show_code:
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="details", size=5),
            Layout(name="evaluations", size=10),
            Layout(name="code"),
            Layout(name="footer", size=3),
        )
    else:
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="details", size=5),
            Layout(name="evaluations"),
            Layout(name="footer", size=3),
        )

    layout["header"].update(
        Panel(
            Text("SUBMISSION DETAILS", justify="center", style="bold white"),
            border_style="white",
        )
    )

    layout["details"].update(create_submission_header(detail))
    layout["evaluations"].update(create_evaluations_table(detail))

    if show_code:
        layout["code"].update(create_code_panel(detail))

    footer_text = "[cyan]c[/] Toggle code  [cyan]r[/] Refresh  [cyan]b/Esc[/] Back  [cyan]q[/] Quit"
    layout["footer"].update(
        Panel(Text.from_markup(footer_text, justify="center"), border_style="dim")
    )

    return layout


def get_key_nonblocking(timeout: float = 0.1) -> str | None:
    """Get a single keypress with timeout (Unix only)."""
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        if sys.stdin in select.select([sys.stdin], [], [], timeout)[0]:
            ch = sys.stdin.read(1)
            # Handle escape sequences (arrow keys)
            if ch == "\x1b":
                if sys.stdin in select.select([sys.stdin], [], [], 0.05)[0]:
                    ch += sys.stdin.read(1)
                    if ch == "\x1b[" and sys.stdin in select.select(
                        [sys.stdin], [], [], 0.05
                    )[0]:
                        ch += sys.stdin.read(1)
            return ch
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def run_tui(base_url: str, refresh_interval: int, demo: bool = False):
    """Run the interactive TUI."""
    import termios

    # Save terminal state
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    # Select client based on demo mode
    client_class = MockClient if demo else TournamentClient
    client_args = () if demo else (base_url,)

    try:
        with client_class(*client_args) as client:
            # State
            data = client.fetch_all()
            current_view = "dashboard"
            current_detail: SubmissionDetail | None = None
            show_code = True
            last_refresh = time.time()

            # Dashboard navigation state
            active_panel = "leaderboard"  # or "recent"
            leaderboard_idx = 0 if data.leaderboard else None
            recent_idx = 0 if data.recent else None

            while True:
                console.clear()

                if current_view == "dashboard":
                    layout = create_dashboard_layout(
                        data,
                        leaderboard_idx=leaderboard_idx if active_panel == "leaderboard" else None,
                        recent_idx=recent_idx if active_panel == "recent" else None,
                        active_panel=active_panel,
                        demo=demo,
                    )
                    console.print(layout)
                elif current_view == "submission" and current_detail:
                    layout = create_submission_layout(current_detail, show_code=show_code)
                    console.print(layout)

                # Handle input
                key = get_key_nonblocking(0.3)

                if key:
                    # Quit
                    if key == "q" or key == "\x03":
                        break

                    elif current_view == "dashboard":
                        # Refresh
                        if key == "r":
                            data = client.fetch_all()
                            if data.leaderboard and leaderboard_idx is None:
                                leaderboard_idx = 0
                            if data.recent and recent_idx is None:
                                recent_idx = 0
                            last_refresh = time.time()

                        # Switch panels with left/right arrows or Tab
                        elif key in ("\x1b[D", "\x1b[C", "\t"):  # Left, Right, Tab
                            if active_panel == "leaderboard":
                                active_panel = "recent"
                            else:
                                active_panel = "leaderboard"

                        # Navigate up
                        elif key == "\x1b[A":  # Up arrow
                            if active_panel == "leaderboard" and leaderboard_idx is not None:
                                leaderboard_idx = max(0, leaderboard_idx - 1)
                            elif active_panel == "recent" and recent_idx is not None:
                                recent_idx = max(0, recent_idx - 1)

                        # Navigate down
                        elif key == "\x1b[B":  # Down arrow
                            if active_panel == "leaderboard" and leaderboard_idx is not None:
                                max_idx = min(9, len(data.leaderboard) - 1)
                                leaderboard_idx = min(max_idx, leaderboard_idx + 1)
                            elif active_panel == "recent" and recent_idx is not None:
                                max_idx = min(9, len(data.recent) - 1)
                                recent_idx = min(max_idx, recent_idx + 1)

                        # Select with Enter
                        elif key in ("\r", "\n"):
                            submission_id = None
                            if active_panel == "leaderboard" and leaderboard_idx is not None:
                                if leaderboard_idx < len(data.leaderboard):
                                    submission_id = data.leaderboard[leaderboard_idx].get(
                                        "submission_id"
                                    )
                            elif active_panel == "recent" and recent_idx is not None:
                                if recent_idx < len(data.recent):
                                    submission_id = data.recent[recent_idx].get("submission_id")

                            if submission_id:
                                current_detail = client.fetch_submission_detail(submission_id)
                                current_view = "submission"
                                show_code = True

                        # Quick select with number keys
                        elif key.isdigit() and key != "0":
                            idx = int(key) - 1
                            if active_panel == "leaderboard":
                                if idx < len(data.leaderboard):
                                    leaderboard_idx = idx
                                    submission_id = data.leaderboard[idx].get("submission_id")
                                    if submission_id:
                                        current_detail = client.fetch_submission_detail(
                                            submission_id
                                        )
                                        current_view = "submission"
                                        show_code = True
                            else:
                                if idx < len(data.recent):
                                    recent_idx = idx
                                    submission_id = data.recent[idx].get("submission_id")
                                    if submission_id:
                                        current_detail = client.fetch_submission_detail(
                                            submission_id
                                        )
                                        current_view = "submission"
                                        show_code = True

                    elif current_view == "submission":
                        # Back to dashboard
                        if key in ("b", "\x1b", "\x1b["):  # b, Esc
                            current_view = "dashboard"
                            current_detail = None

                        # Toggle code
                        elif key == "c":
                            show_code = not show_code

                        # Refresh submission
                        elif key == "r" and current_detail:
                            submission_id = current_detail.submission.get("submission_id")
                            if submission_id:
                                current_detail = client.fetch_submission_detail(submission_id)

                # Auto-refresh dashboard
                if current_view == "dashboard" and time.time() - last_refresh > refresh_interval:
                    data = client.fetch_all()
                    last_refresh = time.time()

    except KeyboardInterrupt:
        pass
    finally:
        # Restore terminal
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        console.clear()


def main():
    """Main entry point for the TUI."""
    parser = argparse.ArgumentParser(
        prog="tplr",
        description="τemplar crusades",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls:
  Dashboard:
    ↑/↓         Navigate entries
    ←/→/Tab     Switch between Leaderboard and Recent
    Enter/1-9   View submission details
    r           Refresh data
    q           Quit

  Submission Detail:
    c           Toggle code view
    r           Refresh
    b/Esc       Back to dashboard
    q           Quit

Examples:
  tplr --demo           # Run with mock data for demo
  tplr --url http://..  # Connect to custom API
        """,
    )

    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Tournament API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=30,
        help="Auto-refresh interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with mock data for demonstration",
    )

    args = parser.parse_args()
    run_tui(args.url, args.refresh, demo=args.demo)


if __name__ == "__main__":
    main()
