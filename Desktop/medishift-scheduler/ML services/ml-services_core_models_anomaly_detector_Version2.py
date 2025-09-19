import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd

class ScheduleAnomalyDetector:
    """
    Detects scheduling anomalies and potential issues
    """
    def __init__(self):
        self.model = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def detect_anomalies(self, schedule_data):
        """
        Detect anomalies in schedule
        """
        features = self.extract_schedule_features(schedule_data)
        scaled_features = self.scaler.fit_transform(features)
        
        # Predict anomalies (-1 for anomaly, 1 for normal)
        predictions = self.model.fit_predict(scaled_features)
        anomaly_scores = self.model.score_samples(scaled_features)
        
        anomalies = []
        for i, (pred, score) in enumerate(zip(predictions, anomaly_scores)):
            if pred == -1:
                anomalies.append({
                    'index': i,
                    'type': self.classify_anomaly(features.iloc[i]),
                    'severity': self.calculate_severity(score),
                    'details': self.get_anomaly_details(features.iloc[i], schedule_data[i]),
                    'recommendation': self.get_recommendation(features.iloc[i])
                })
        
        return {
            'total_anomalies': len(anomalies),
            'anomalies': anomalies,
            'health_score': self.calculate_health_score(predictions)
        }
    
    def extract_schedule_features(self, schedule_data):
        """
        Extract features from schedule for anomaly detection
        """
        features = []
        
        for entry in schedule_data:
            features.append({
                'calls_per_resident': entry.get('calls_count', 0),
                'consecutive_calls': entry.get('consecutive_calls', 0),
                'weekend_concentration': entry.get('weekend_ratio', 0),
                'coverage_gaps': entry.get('coverage_gaps', 0),
                'fairness_index': entry.get('fairness_index', 1),
                'pgy_distribution': entry.get('pgy_variance', 0),
                'vacation_conflicts': entry.get('vacation_conflicts', 0),
                'workload_variance': entry.get('workload_variance', 0)
            })
        
        return pd.DataFrame(features)
    
    def classify_anomaly(self, features):
        """
        Classify type of anomaly
        """
        if features['consecutive_calls'] > 2:
            return 'Excessive Consecutive Calls'
        elif features['coverage_gaps'] > 0:
            return 'Coverage Gap Detected'
        elif features['fairness_index'] < 0.5:
            return 'Unfair Distribution'
        elif features['vacation_conflicts'] > 0:
            return 'Vacation Conflict'
        else:
            return 'General Anomaly'
    
    def calculate_severity(self, anomaly_score):
        """
        Calculate severity of anomaly
        """
        if anomaly_score < -0.5:
            return 'Critical'
        elif anomaly_score < -0.3:
            return 'High'
        elif anomaly_score < -0.1:
            return 'Medium'
        else:
            return 'Low'
    
    def get_anomaly_details(self, features, original_data):
        """
        Get detailed information about anomaly
        """
        return {
            'affected_residents': original_data.get('residents', []),
            'date_range': original_data.get('date_range', ''),
            'metrics': features.to_dict()
        }
    
    def get_recommendation(self, features):
        """
        Generate recommendation for fixing anomaly
        """
        if features['consecutive_calls'] > 2:
            return 'Redistribute calls to prevent burnout. Consider adding post-call days.'
        elif features['coverage_gaps'] > 0:
            return 'Fill coverage gaps by adjusting vacation approvals or adding backup residents.'
        elif features['fairness_index'] < 0.5:
            return 'Rebalance call distribution to ensure fairness across all residents.'
        elif features['vacation_conflicts'] > 0:
            return 'Review and adjust vacation schedules to avoid conflicts.'
        else:
            return 'Review schedule manually for optimization opportunities.'
    
    def calculate_health_score(self, predictions):
        """
        Calculate overall schedule health score
        """
        normal_count = np.sum(predictions == 1)
        total_count = len(predictions)
        
        return (normal_count / total_count) * 100 if total_count > 0 else 0