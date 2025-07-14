import pandas as pd
import matplotlib.pyplot as plt
from database.session import SessionLocal
from database.models import Trade


def load_trades():
    db = SessionLocal()
    trades = db.query(Trade).order_by(Trade.timestamp).all()
    data = []
    for t in trades:
        pnl = (t.take_profit - t.entry_price) * t.size * (1 if t.order_type == 'BUY' else -1)
        data.append({'timestamp': t.timestamp, 'pnl': pnl})
    df = pd.DataFrame(data)
    if df.empty:
        print("No trades found.")
        return df
    df['cum_pnl'] = df['pnl'].cumsum()
    return df


def plot_cum_pnl():
    df = load_trades()
    if df.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(df['timestamp'], df['cum_pnl'], marker='o')
    plt.title('Cumulative P&L')
    plt.xlabel('Time')
    plt.ylabel('P&L')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    plot_cum_pnl()
