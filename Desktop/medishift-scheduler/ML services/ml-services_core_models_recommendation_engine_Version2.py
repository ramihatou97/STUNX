import numpy as np
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime, timedelta

class SmartScheduleRecommender:
    """
    Provides intelligent scheduling recommendations
    """
    def __init__(self):
        self.rules = self.load_scheduling_rules()
        self.preferences = {}
        
    def generate_recommendations(self, current_schedule, constraints):
        """
        Generate smart recommendations for schedule optimization
        """
        recommendations = []
        
        # Analyze current schedule
        analysis = self.analyze_schedule(current_schedule)
        
        # Generate recommendations based on analysis
        if analysis['workload_imbalance'] > 0.2:
            recommendations.append(self.recommend_workload_balance(current_schedule))
        
        if analysis['coverage_gaps']:
            recommendations.append(self.recommend_coverage_fixes(analysis['coverage_gaps']))
        
        if analysis['vacation_conflicts']:
            recommendations.append(self.recommend_vacation_solutions(analysis['vacation_conflicts']))
        
        if analysis['fairness_score'] < 0.7:
            recommendations.append(self.recommend_fairness_improvements(current_schedule))
        
        # Add predictive recommendations
        predictive_recs = self.generate_predictive_recommendations(current_schedule)
        recommendations.extend(predictive_recs)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis,
            'recommendations': recommendations,
            'priority_actions': self.prioritize_recommendations(recommendations),
            'estimated_improvement': self.estimate_improvement(recommendations)
        }
    
    def analyze_schedule(self, schedule):
        """
        Comprehensive schedule analysis
        """
        return {
            'workload_imbalance': self.calculate_workload_imbalance(schedule),
            'coverage_gaps': self.find_coverage_gaps(schedule),
            'vacation_conflicts': self.find_vacation_conflicts(schedule),
            'fairness_score': self.calculate_fairness_score(schedule),
            'efficiency_score': self.calculate_efficiency_score(schedule),
            'compliance_issues': self.check_compliance(schedule)
        }
    
    def recommend_workload_balance(self, schedule):
        """
        Recommend workload balancing changes
        """
        overloaded = []
        underloaded = []
        
        for resident in schedule['residents']:
            if resident['call_count'] > schedule['average_calls'] * 1.2:
                overloaded.append(resident)
            elif resident['call_count'] < schedule['average_calls'] * 0.8:
                underloaded.append(resident)
        
        swaps = []
        for over in overloaded:
            for under in underloaded:
                if self.can_swap(over, under, schedule):
                    swaps.append({
                        'from': over['id'],
                        'to': under['id'],
                        'dates': self.find_swappable_dates(over, under, schedule)
                    })
        
        return {
            'type': 'workload_balance',
            'priority': 'high',
            'description': f'Rebalance workload: {len(overloaded)} overloaded, {len(underloaded)} underloaded residents',
            'suggested_swaps': swaps[:5],  # Top 5 swaps
            'impact': 'Improves fairness by 25%'
        }
    
    def recommend_coverage_fixes(self, gaps):
        """
        Recommend fixes for coverage gaps
        """
        fixes = []
        
        for gap in gaps:
            available_residents = self.find_available_residents(gap['date'])
            
            fixes.append({
                'gap_date': gap['date'],
                'type': gap['type'],
                'available_options': [
                    {
                        'resident_id': r['id'],
                        'resident_name': r['name'],
                        'suitability_score': self.calculate_suitability(r, gap)
                    }
                    for r in available_residents[:3]
                ]
            })
        
        return {
            'type': 'coverage_gaps',
            'priority': 'critical',
            'description': f'Fix {len(gaps)} coverage gaps',
            'fixes': fixes,
            'impact': 'Ensures 100% coverage compliance'
        }
    
    def recommend_vacation_solutions(self, conflicts):
        """
        Recommend solutions for vacation conflicts
        """
        solutions = []
        
        for conflict in conflicts:
            solutions.append({
                'conflict': conflict,
                'options': [
                    {
                        'type': 'reschedule',
                        'suggestion': f"Move vacation to {self.find_better_vacation_dates(conflict)}"
                    },
                    {
                        'type': 'coverage',
                        'suggestion': f"Assign backup: {self.find_backup_resident(conflict)}"
                    },
                    {
                        'type': 'split',
                        'suggestion': 'Split vacation into two periods to maintain coverage'
                    }
                ]
            })
        
        return {
            'type': 'vacation_conflicts',
            'priority': 'medium',
            'description': f'Resolve {len(conflicts)} vacation conflicts',
            'solutions': solutions,
            'impact': 'Maintains coverage while honoring leave requests'
        }
    
    def recommend_fairness_improvements(self, schedule):
        """
        Recommend fairness improvements
        """
        return {
            'type': 'fairness',
            'priority': 'medium',
            'description': 'Improve schedule fairness',
            'suggestions': [
                'Implement rotation-based weekend assignments',
                'Balance holiday coverage across all PGY levels',
                'Ensure equal distribution of prime vacation slots',
                'Review and adjust call points system'
            ],
            'impact': 'Increases satisfaction and reduces burnout'
        }
    
    def generate_predictive_recommendations(self, schedule):
        """
        Generate forward-looking recommendations
        """
        predictions = []
        
        # Predict high-demand periods
        high_demand = self.predict_high_demand_periods()
        if high_demand:
            predictions.append({
                'type': 'predictive',
                'priority': 'low',
                'description': 'Prepare for upcoming high-demand periods',
                'details': high_demand,
                'impact': 'Proactive planning reduces last-minute changes'
            })
        
        # Predict potential conflicts
        future_conflicts = self.predict_future_conflicts(schedule)
        if future_conflicts:
            predictions.append({
                'type': 'predictive',
                'priority': 'medium',
                'description': 'Potential future conflicts detected',
                'details': future_conflicts,
                'impact': 'Early intervention prevents scheduling issues'
            })
        
        return predictions
    
    def prioritize_recommendations(self, recommendations):
        """
        Prioritize recommendations by impact and urgency
        """
        priority_map = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        
        sorted_recs = sorted(
            recommendations,
            key=lambda x: priority_map.get(x['priority'], 4)
        )
        
        return sorted_recs[:3]  # Top 3 priority actions
    
    def estimate_improvement(self, recommendations):
        """
        Estimate overall improvement from recommendations
        """
        improvements = {
            'coverage': 0,
            'fairness': 0,
            'efficiency': 0,
            'satisfaction': 0
        }
        
        for rec in recommendations:
            if rec['type'] == 'coverage_gaps':
                improvements['coverage'] += 30
            elif rec['type'] == 'workload_balance':
                improvements['fairness'] += 25
                improvements['satisfaction'] += 20
            elif rec['type'] == 'vacation_conflicts':
                improvements['satisfaction'] += 15
            elif rec['type'] == 'fairness':
                improvements['fairness'] += 20
                improvements['satisfaction'] += 25
        
        return {
            metric: min(100, value) 
            for metric, value in improvements.items()
        }
    
    # Helper methods
    def load_scheduling_rules(self):
        return {
            'max_consecutive_calls': 2,
            'min_rest_between_calls': 2,
            'max_weekend_calls': 2,
            'fairness_threshold': 0.7
        }
    
    def calculate_workload_imbalance(self, schedule):
        calls = [r['call_count'] for r in schedule.get('residents', [])]
        if not calls:
            return 0
        return np.std(calls) / np.mean(calls) if np.mean(calls) > 0 else 0
    
    def find_coverage_gaps(self, schedule):
        # Simplified - would check actual schedule data
        return []
    
    def find_vacation_conflicts(self, schedule):
        # Simplified - would check actual conflicts
        return []
    
    def calculate_fairness_score(self, schedule):
        # Gini coefficient calculation
        return 0.75  # Placeholder
    
    def calculate_efficiency_score(self, schedule):
        return 0.80  # Placeholder
    
    def check_compliance(self, schedule):
        return []  # Placeholder