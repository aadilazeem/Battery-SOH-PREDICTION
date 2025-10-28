# maintenance_scheduler.py

def calculate_remaining_useful_life(current_soh, fade_rate):
    """
    Calculate RUL (Remaining Useful Life)
    
    EOL Threshold: 0.80 (80% capacity)
    RUL = (Current_SOH - EOL) / Fade_Rate
    """
    eol_threshold = 0.80
    
    if current_soh <= eol_threshold:
        return 0  # Already EOL
    
    remaining_capacity = current_soh - eol_threshold
    rul_cycles = remaining_capacity / fade_rate
    
    # Estimate time (assume 5 cycles/day)
    cycles_per_day = 5
    rul_days = rul_cycles / cycles_per_day
    
    return {
        'rul_cycles': int(rul_cycles),
        'rul_days': int(rul_days),
        'rul_months': int(rul_days / 30),
        'maintenance_date': datetime.now() + timedelta(days=rul_days)
    }

def generate_maintenance_schedule(battery_fleet):
    """
    Generate proactive maintenance schedule
    
    Rules:
    - Schedule maintenance 30 days before EOL
    - Prioritize batteries reaching 70% SOH
    - Group maintenance by location for efficiency
    """
    schedule = []
    
    for battery in battery_fleet:
        soh = battery['current_soh']
        fade_rate = battery['fade_rate']
        rul = calculate_remaining_useful_life(soh, fade_rate)
        
        # Maintenance urgency
        if soh < 0.70:
            urgency = "CRITICAL - Replace immediately"
            priority = 1
        elif soh < 0.75:
            urgency = "HIGH - Schedule within 2 weeks"
            priority = 2
        elif rul['rul_days'] < 30:
            urgency = "MEDIUM - Schedule next month"
            priority = 3
        else:
            urgency = "LOW - Monitor"
            priority = 4
        
        schedule.append({
            'battery_id': battery['id'],
            'current_soh': soh,
            'rul_days': rul['rul_days'],
            'maintenance_date': rul['maintenance_date'],
            'urgency': urgency,
            'priority': priority,
            'estimated_cost': calculate_replacement_cost(soh)
        })
    
    # Sort by priority and RUL
    schedule = sorted(schedule, key=lambda x: (x['priority'], x['rul_days']))
    
    return schedule

# Usage
battery_fleet = [
    {'id': 'B001', 'current_soh': 0.85, 'fade_rate': 0.0012},
    {'id': 'B002', 'current_soh': 0.72, 'fade_rate': 0.0015},
    {'id': 'B003', 'current_soh': 0.68, 'fade_rate': 0.0020},
]

maintenance_plan = generate_maintenance_schedule(battery_fleet)

for item in maintenance_plan:
    print(f"""
    Battery: {item['battery_id']}
    Current SOH: {item['current_soh']:.1%}
    RUL: {item['rul_days']} days ({item['maintenance_date'].strftime('%Y-%m-%d')})
    Urgency: {item['urgency']}
    Priority: {item['priority']}/4
    Est. Cost: ${item['estimated_cost']}
    """)