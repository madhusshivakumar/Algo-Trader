"""Live trading dashboard — web UI for monitoring the bot."""

import sqlite3
import json
from datetime import datetime
from flask import Flask, jsonify, render_template_string
from core.broker import Broker
from config import Config

app = Flask(__name__)
DB_PATH = "trades.db"


def get_broker():
    return Broker()


def query_db(sql, args=()):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, args).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.route("/api/account")
def api_account():
    broker = get_broker()
    account = broker.get_account()
    positions = broker.get_positions()
    return jsonify({"account": account, "positions": positions})


@app.route("/api/trades")
def api_trades():
    trades = query_db("SELECT * FROM trades ORDER BY id DESC LIMIT 50")
    return jsonify(trades)


@app.route("/api/equity")
def api_equity():
    snapshots = query_db("SELECT * FROM equity_snapshots ORDER BY id ASC")
    return jsonify(snapshots)


@app.route("/api/stats")
def api_stats():
    trades = query_db("SELECT * FROM trades")
    buys = [t for t in trades if t["side"] == "buy"]
    sells = [t for t in trades if t["side"] == "sell"]
    wins = [t for t in sells if (t.get("pnl") or 0) > 0]
    losses = [t for t in sells if (t.get("pnl") or 0) < 0]
    total_pnl = sum(t.get("pnl") or 0 for t in trades)
    win_rate = (len(wins) / len(sells) * 100) if sells else 0
    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0

    return jsonify({
        "total_trades": len(buys),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
    })


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Algo Trader Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text-dim: #8b949e; --green: #3fb950;
    --red: #f85149; --blue: #58a6ff; --yellow: #d29922;
    --purple: #bc8cff;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: -apple-system, 'SF Mono', 'Fira Code', monospace; padding: 20px; }
  h1 { font-size: 1.4rem; margin-bottom: 4px; }
  .subtitle { color: var(--text-dim); font-size: 0.85rem; margin-bottom: 20px; }
  .live-dot { display: inline-block; width: 8px; height: 8px; background: var(--green); border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite; }
  @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
  .card-label { color: var(--text-dim); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
  .card-value { font-size: 1.8rem; font-weight: 700; }
  .card-value.green { color: var(--green); }
  .card-value.red { color: var(--red); }
  .card-sub { color: var(--text-dim); font-size: 0.8rem; margin-top: 4px; }

  .chart-container { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 24px; }
  .chart-title { font-size: 0.9rem; font-weight: 600; margin-bottom: 12px; }

  .positions-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .position-card { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
  .position-card .symbol { font-size: 1.2rem; font-weight: 700; margin-bottom: 8px; }
  .position-card .row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 0.85rem; }
  .position-card .row .label { color: var(--text-dim); }

  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; color: var(--text-dim); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; padding: 10px 12px; border-bottom: 1px solid var(--border); }
  td { padding: 10px 12px; border-bottom: 1px solid var(--border); font-size: 0.85rem; }
  tr:hover { background: rgba(88, 166, 255, 0.05); }
  .badge { padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
  .badge-buy { background: rgba(63, 185, 80, 0.15); color: var(--green); }
  .badge-sell { background: rgba(248, 81, 73, 0.15); color: var(--red); }
  .pnl-pos { color: var(--green); }
  .pnl-neg { color: var(--red); }
  .no-data { text-align: center; color: var(--text-dim); padding: 40px; }
  .updated { color: var(--text-dim); font-size: 0.75rem; text-align: right; margin-top: 12px; }
  .section-title { font-size: 1rem; font-weight: 600; margin-bottom: 12px; }

  .stats-bar { display: flex; gap: 24px; margin-bottom: 24px; flex-wrap: wrap; }
  .stat { display: flex; align-items: center; gap: 6px; font-size: 0.85rem; }
  .stat .dot { width: 6px; height: 6px; border-radius: 50%; }
</style>
</head>
<body>

<div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px;">
  <div>
    <h1><span class="live-dot"></span> Algo Trader Dashboard</h1>
    <div class="subtitle">RSI Divergence Strategy &middot; BTC/USD, ETH/USD &middot; Paper Trading</div>
  </div>
  <div class="updated" id="lastUpdate">Loading...</div>
</div>

<div class="grid" id="statsCards">
  <div class="card"><div class="card-label">Equity</div><div class="card-value" id="equity">—</div><div class="card-sub" id="equityChange"></div></div>
  <div class="card"><div class="card-label">Cash</div><div class="card-value" id="cash">—</div></div>
  <div class="card"><div class="card-label">Total P&L</div><div class="card-value" id="totalPnl">—</div><div class="card-sub" id="pnlTrades"></div></div>
  <div class="card"><div class="card-label">Win Rate</div><div class="card-value" id="winRate">—</div><div class="card-sub" id="winLoss"></div></div>
</div>

<div class="chart-container">
  <div class="chart-title">Equity Curve</div>
  <canvas id="equityChart" height="80"></canvas>
  <div id="noEquityData" class="no-data" style="display:none;">Equity data will appear after the bot runs for a few cycles</div>
</div>

<div class="section-title">Open Positions</div>
<div class="positions-grid" id="positions">
  <div class="no-data">No open positions</div>
</div>

<div class="chart-container" style="margin-top: 24px;">
  <div class="chart-title">P&L per Trade</div>
  <canvas id="pnlChart" height="60"></canvas>
  <div id="noPnlData" class="no-data" style="display:none;">Trade P&L data will appear after completed trades</div>
</div>

<div class="chart-container">
  <div class="section-title">Recent Trades</div>
  <table>
    <thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Amount</th><th>Price</th><th>Reason</th><th>P&L</th></tr></thead>
    <tbody id="tradesBody"><tr><td colspan="7" class="no-data">No trades yet</td></tr></tbody>
  </table>
</div>

<script>
let equityChart = null;
let pnlChart = null;

function fmt(n, decimals=2) { return n != null ? '$' + Number(n).toFixed(decimals) : '—'; }
function fmtPct(n) { return n != null ? Number(n).toFixed(1) + '%' : '—'; }
function pnlClass(n) { return n > 0 ? 'pnl-pos' : n < 0 ? 'pnl-neg' : ''; }
function pnlColor(n) { return n > 0 ? '#3fb950' : n < 0 ? '#f85149' : '#8b949e'; }

async function fetchJSON(url) {
  const res = await fetch(url);
  return res.json();
}

async function refresh() {
  try {
    const [account, stats, trades, equity] = await Promise.all([
      fetchJSON('/api/account'),
      fetchJSON('/api/stats'),
      fetchJSON('/api/trades'),
      fetchJSON('/api/equity'),
    ]);

    // Stats cards
    document.getElementById('equity').textContent = fmt(account.account.equity);
    document.getElementById('cash').textContent = fmt(account.account.cash);

    const pnlEl = document.getElementById('totalPnl');
    pnlEl.textContent = fmt(stats.total_pnl);
    pnlEl.className = 'card-value ' + (stats.total_pnl >= 0 ? 'green' : 'red');
    document.getElementById('pnlTrades').textContent = stats.total_trades + ' trades';

    const wrEl = document.getElementById('winRate');
    wrEl.textContent = fmtPct(stats.win_rate);
    wrEl.className = 'card-value ' + (stats.win_rate >= 50 ? 'green' : stats.win_rate > 0 ? 'red' : '');
    document.getElementById('winLoss').textContent = stats.wins + 'W / ' + stats.losses + 'L';

    // Positions
    const posDiv = document.getElementById('positions');
    if (account.positions.length === 0) {
      posDiv.innerHTML = '<div class="no-data">No open positions</div>';
    } else {
      posDiv.innerHTML = account.positions.map(p => `
        <div class="position-card">
          <div class="symbol">${p.symbol}</div>
          <div class="row"><span class="label">Quantity</span><span>${Number(p.qty).toFixed(6)}</span></div>
          <div class="row"><span class="label">Entry Price</span><span>${fmt(p.avg_entry_price)}</span></div>
          <div class="row"><span class="label">Current Price</span><span>${fmt(p.current_price)}</span></div>
          <div class="row"><span class="label">Market Value</span><span>${fmt(p.market_value)}</span></div>
          <div class="row"><span class="label">Unrealized P&L</span><span class="${pnlClass(p.unrealized_pl)}">${fmt(p.unrealized_pl)} (${(p.unrealized_plpc * 100).toFixed(2)}%)</span></div>
        </div>
      `).join('');
    }

    // Trades table
    const tbody = document.getElementById('tradesBody');
    if (trades.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" class="no-data">No trades yet — bot is scanning for signals...</td></tr>';
    } else {
      tbody.innerHTML = trades.map(t => `
        <tr>
          <td style="color:var(--text-dim)">${t.timestamp}</td>
          <td><strong>${t.symbol}</strong></td>
          <td><span class="badge badge-${t.side}">${t.side.toUpperCase()}</span></td>
          <td>${fmt(t.amount)}</td>
          <td>${fmt(t.price)}</td>
          <td style="color:var(--text-dim);max-width:200px;overflow:hidden;text-overflow:ellipsis">${t.reason || '—'}</td>
          <td class="${pnlClass(t.pnl)}">${t.pnl ? fmt(t.pnl) : '—'}</td>
        </tr>
      `).join('');
    }

    // Equity chart
    if (equity.length > 1) {
      document.getElementById('noEquityData').style.display = 'none';
      const labels = equity.map(e => e.timestamp);
      const data = equity.map(e => e.equity);
      if (equityChart) {
        equityChart.data.labels = labels;
        equityChart.data.datasets[0].data = data;
        equityChart.update('none');
      } else {
        equityChart = new Chart(document.getElementById('equityChart'), {
          type: 'line',
          data: {
            labels,
            datasets: [{
              data,
              borderColor: '#58a6ff',
              backgroundColor: 'rgba(88,166,255,0.1)',
              fill: true,
              tension: 0.3,
              pointRadius: 0,
              borderWidth: 2,
            }]
          },
          options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
              x: { display: true, ticks: { color: '#8b949e', maxTicksLimit: 8, font: { size: 10 } }, grid: { color: '#21262d' } },
              y: { ticks: { color: '#8b949e', callback: v => '$' + v.toFixed(0) }, grid: { color: '#21262d' } }
            }
          }
        });
      }
    } else {
      document.getElementById('noEquityData').style.display = 'block';
    }

    // PnL chart
    const sellTrades = trades.filter(t => t.side === 'sell' && t.pnl != null).reverse();
    if (sellTrades.length > 0) {
      document.getElementById('noPnlData').style.display = 'none';
      const pnlLabels = sellTrades.map((t, i) => '#' + (i + 1) + ' ' + t.symbol);
      const pnlData = sellTrades.map(t => t.pnl);
      const pnlColors = pnlData.map(v => pnlColor(v));
      if (pnlChart) {
        pnlChart.data.labels = pnlLabels;
        pnlChart.data.datasets[0].data = pnlData;
        pnlChart.data.datasets[0].backgroundColor = pnlColors;
        pnlChart.update('none');
      } else {
        pnlChart = new Chart(document.getElementById('pnlChart'), {
          type: 'bar',
          data: {
            labels: pnlLabels,
            datasets: [{
              data: pnlData,
              backgroundColor: pnlColors,
              borderRadius: 4,
            }]
          },
          options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
              x: { ticks: { color: '#8b949e', font: { size: 10 } }, grid: { display: false } },
              y: { ticks: { color: '#8b949e', callback: v => '$' + v.toFixed(2) }, grid: { color: '#21262d' } }
            }
          }
        });
      }
    } else {
      document.getElementById('noPnlData').style.display = 'block';
    }

    document.getElementById('lastUpdate').textContent = 'Updated: ' + new Date().toLocaleTimeString();
  } catch (err) {
    console.error('Refresh failed:', err);
  }
}

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("\n  Dashboard: http://localhost:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
