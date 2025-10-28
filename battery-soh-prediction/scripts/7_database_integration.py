# database_integration.py - Store predictions in database

from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.orm import sessionmaker
import sqlalchemy.ext.declarative as declarative

Base = declarative.declarative_base()

class BatteryPrediction(Base):
    __tablename__ = 'battery_predictions'
    
    id = Column(String(50), primary_key=True)
    battery_id = Column(String(50), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    soh_actual = Column(Float)
    soh_predicted = Column(Float)
    confidence_interval_lower = Column(Float)
    confidence_interval_upper = Column(Float)
    error = Column(Float)
    prediction_status = Column(String(20))
    model_version = Column(String(20))

# Database setup
engine = create_engine('postgresql://user:password@localhost/battery_db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

import datetime
# Store predictions
session = Session()
prediction = BatteryPrediction(
    id='pred_001',
    battery_id='B001',
    timestamp=datetime.now(),
    soh_predicted=0.8524,
    confidence_interval_lower=0.8488,
    confidence_interval_upper=0.8560,
    prediction_status='OK',
    model_version='v1.0'
)
session.add(prediction)
session.commit()

# Query predictions
recent_predictions = session.query(BatteryPrediction)\
    .filter(BatteryPrediction.battery_id == 'B001')\
    .order_by(BatteryPrediction.timestamp.desc())\
    .limit(10)\
    .all()