import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import firestore

class CallLoadPredictor:
    """
    Predicts future call load and optimal distribution
    """
    def __init__(self, model_path='models/schedule_predictor.pkl'):
        self.model = self.load_model(model_path)
        self.scaler = StandardScaler()
        self.db = firestore.client()
        
    def load_model(self, path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except:
            # Initialize new model if not found
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
    
    def extract_features(self, date, resident_data, historical_data):
        """
        Extract features for prediction
        """
        features = {
            'day_of_week': date.weekday(),
            'month': date.month,
            'is_weekend': date.weekday() >= 5,
            'is_holiday': self.is_holiday(date),
            'days_since_last_call': self.calculate_days_since_last_call(resident_data),
            'total_calls_this_month': resident_data.get('monthly_calls', 0),
            'pgy_level': resident_data.get('pgy_level', 1),
            'historical_avg_calls': self.calculate_historical_average(historical_data),
            'team_size': resident_data.get('team_size', 10),
            'vacation_conflicts': self.check_vacation_conflicts(date),
        }
        return pd.DataFrame([features])
    
    def predict_call_load(self, date, period_days=30):
        """
        Predict call load for next period
        """
        predictions = []
        current_date = date
        
        for _ in range(period_days):
            # Get resident data
            residents = self.get_active_residents(current_date)
            
            daily_predictions = []
            for resident in residents:
                features = self.extract_features(
                    current_date,
                    resident,
                    self.get_historical_data(resident['id'])
                )
                
                # Predict probability of needing call coverage
                scaled_features = self.scaler.fit_transform(features)
                probability = self.model.predict_proba(scaled_features)[0][1]
                
                daily_predictions.append({
                    'date': current_date,
                    'resident_id': resident['id'],
                    'resident_name': resident['name'],
                    'call_probability': probability,
                    'recommended': probability > 0.7
                })
            
            predictions.append({
                'date': current_date.isoformat(),
                'total_coverage_needed': sum(1 for p in daily_predictions if p['recommended']),
                'assignments': sorted(daily_predictions, key=lambda x: x['call_probability'], reverse=True)
            })
            
            current_date += timedelta(days=1)
        
        return {
            'period_start': date.isoformat(),
            'period_end': (date + timedelta(days=period_days-1)).isoformat(),
            'predictions': predictions,
            'confidence': self.calculate_confidence_score(predictions)
        }
    
    def calculate_confidence_score(self, predictions):
        """
        Calculate overall confidence in predictions
        """
        # Simplified confidence calculation
        avg_probability = np.mean([
            p['call_probability'] 
            for day in predictions 
            for p in day['assignments']
        ])
        return min(0.95, avg_probability + 0.3)
    
    def get_active_residents(self, date):
        """
        Get residents available on given date
        """
        # Fetch from Firestore
        residents_ref = self.db.collection('residents')
        residents = residents_ref.where('active', '==', True).get()
        return [r.to_dict() for r in residents]
    
    def is_holiday(self, date):
        """
        Check if date is a holiday
        """
        holidays = self.db.collection('holidays').get()
        holiday_dates = [h.to_dict()['date'].date() for h in holidays]
        return date.date() in holiday_dates
    
    def calculate_days_since_last_call(self, resident_data):
        """
        Calculate days since resident's last call
        """
        last_call = resident_data.get('last_call_date')
        if not last_call:
            return 30  # Default if no previous calls
        
        return (datetime.now() - last_call).days
    
    def check_vacation_conflicts(self, date):
        """
        Check for vacation conflicts on date
        """
        vacations = self.db.collection('leaveRequests')\
            .where('status', '==', 'Approved')\
            .where('startDate', '<=', date)\
            .where('endDate', '>=', date)\
            .get()
        
        return len(vacations)
    
    def get_historical_data(self, resident_id):
        """
        Get historical call data for resident
        """
        calls = self.db.collection('callAssignments')\
            .where('residentId', '==', resident_id)\
            .order_by('date', direction=firestore.Query.DESCENDING)\
            .limit(100)\
            .get()
        
        return [c.to_dict() for c in calls]
    
    def train_model(self, training_data):
        """
        Train the prediction model
        """
        X = training_data.drop(['target'], axis=1)
        y = training_data['target']
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # Save model
        with open('models/schedule_predictor.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        return {
            'status': 'trained',
            'accuracy': self.model.score(X_scaled, y),
            'features': list(X.columns)
        }