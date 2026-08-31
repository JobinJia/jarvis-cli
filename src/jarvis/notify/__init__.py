"""Optional remote/webhook notification layer.

The daemon's primary output is local audio; this package adds an opt-in
fire-and-forget push of the spoken line + event metadata to a configurable
webhook (Bark / ntfy / Slack / Discord / any generic POST endpoint) so a
phone or IM can surface it when the user is away from the machine.
"""
