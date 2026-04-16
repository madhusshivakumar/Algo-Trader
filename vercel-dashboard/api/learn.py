"""GET /api/learn — beginner-friendly mini-lessons for dashboard /learn page."""

from http.server import BaseHTTPRequestHandler
import json


_LESSONS = [
    {
        "title": "What is a trading strategy?",
        "body": (
            "A trading strategy is a set of rules that tells the bot when to "
            "buy and sell. This bot runs 9 different strategies — some look for "
            "stocks that dropped too far (mean reversion), others ride upward "
            "trends (momentum). The bot picks the best one per symbol."
        ),
    },
    {
        "title": "What does the RL selector do?",
        "body": (
            "RL stands for Reinforcement Learning. The bot trains a small AI "
            "model that learns which strategy works best for each symbol in the "
            "current market environment. It's like a coach that picks the "
            "starting lineup based on past performance and today's conditions."
        ),
    },
    {
        "title": "Why does the bot pick these symbols?",
        "body": (
            "The Market Scanner agent runs daily, scoring hundreds of symbols "
            "on volume, volatility, and signal quality. It picks the symbols "
            "where the bot's strategies historically perform best. High-volume "
            "stocks with moderate volatility tend to produce cleaner signals."
        ),
    },
    {
        "title": "What is a trailing stop?",
        "body": (
            "A trailing stop automatically sells a position if its price drops "
            "below a threshold. As the price rises, the stop rises too — "
            "locking in gains. If the price falls, the stop triggers a sell. "
            "Think of it as an automatic safety net."
        ),
    },
    {
        "title": "What is a market regime?",
        "body": (
            "The bot classifies the market into four modes: calm (low vol), "
            "normal, volatile (high vol), and crisis. Different strategies "
            "work in different regimes — mean reversion fails in a crash, "
            "momentum fails in a calm market. The bot adapts automatically."
        ),
    },
    {
        "title": "Why does the bot sometimes do nothing?",
        "body": (
            "Holding is a valid action. If no strategy produces a strong "
            "enough signal, the bot waits. Over-trading — buying and selling "
            "too often — eats into profits through transaction costs. The bot "
            "has minimum signal strength and daily trade limits to prevent this."
        ),
    },
    {
        "title": "Can I lose money?",
        "body": (
            "Yes. Most retail algo traders lose money. This bot has safety "
            "rails — daily loss limits, trailing stops, position sizing scaled "
            "to your account — but no algorithm can guarantee profits. Start "
            "with paper trading and small amounts you can afford to lose."
        ),
    },
]


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(_LESSONS).encode())
