from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, Float, String, DateTime

Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    order_type = Column(String, nullable=False)  # BUY or SELL
    size = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    confidence = Column(Float)
    timestamp = Column(DateTime)
