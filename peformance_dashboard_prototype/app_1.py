# app.py - Complete working prototype
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import random
from typing import Dict, List, Optional
import hashlib
import json

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Palmetto Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== TEST DATA GENERATION ====================
def generate_test_data(num_reps: int = 10, num_days: int = 180) -> pd.DataFrame:
    """Generate realistic test sales data"""
    
    # Rep information
    rep_first_names = ["Hunter", "Alex", "Jordan", "Taylor", "Morgan", 
                       "Casey", "Riley", "Quinn", "Dakota", "Skyler",
                       "Jamie", "Cameron", "Drew", "Blake", "Payton"]
    rep_last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones",
                      "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    
    statuses = ['Active', 'Pending', 'Cancelled', 'Churned', 'No Installation Scheduled']
    roles = ['Rep', 'Senior Rep', 'Team Lead']
    
    # Generate rep list
    reps = []
    for i in range(num_reps):
        rep_id = f"REP{i+1:03d}"
        first_name = random.choice(rep_first_names)
        last_name = random.choice(rep_last_names)
        reps.append({
            'rep_id': rep_id,
            'rep_name': f"{first_name} {last_name}",
            'role': random.choice(roles),
            'territory': random.choice(['North', 'South', 'East', 'West', 'Central']),
            'hire_date': datetime.now() - timedelta(days=random.randint(100, 1000))
        })
    
    # Generate sales data
    all_data = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days)
    
    order_id_counter = 1000
    
    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        
        # Each day, generate 0-5 orders per rep
        for rep in reps:
            daily_orders = random.randint(0, 5)
            
            for _ in range(daily_orders):
                order_id = f"ORD{order_id_counter}"
                order_id_counter += 1
                
                # Base order value
                base_value = random.uniform(500, 5000)
                
                # Status probability weights
                status_weights = [0.6, 0.2, 0.1, 0.08, 0.02]  # Active, Pending, Cancelled, Churned, No Install
                status = random.choices(statuses, weights=status_weights)[0]
                
                # Dates
                order_date = current_date + timedelta(hours=random.randint(9, 17))
                
                # Activation date (if not cancelled/churned)
                if status in ['Active', 'Pending']:
                    activation_delay = random.randint(1, 21)  # 1-21 days
                    activation_date = order_date + timedelta(days=activation_delay)
                else:
                    activation_date = None
                
                # Churn date (if churned)
                if status == 'Churned' and activation_date:
                    churn_delay = random.randint(30, 180)
                    churn_date = activation_date + timedelta(days=churn_delay)
                else:
                    churn_date = None
                
                # Week ending (Saturday)
                days_to_saturday = (5 - order_date.weekday()) % 7
                week_ending = order_date + timedelta(days=days_to_saturday)
                
                # Add some seasonality - more sales on weekdays
                if order_date.weekday() < 5:  # Weekday
                    base_value *= random.uniform(1.1, 1.3)
                
                order_data = {
                    'rep_id': rep['rep_id'],
                    'rep_name': rep['rep_name'],
                    'order_id': order_id,
                    'order_date': order_date,
                    'order_value': round(base_value, 2),
                    'status': status,
                    'activation_date': activation_date,
                    'churn_date': churn_date,
                    'week_ending': week_ending,
                    'territory': rep['territory'],
                    'product_type': random.choice(['Residential', 'Commercial', 'Industrial']),
                    'sales_channel': random.choice(['Direct', 'Referral', 'Partner']),
                    'installation_complexity': random.choice(['Simple', 'Standard', 'Complex'])
                }
                
                all_data.append(order_data)
    
    df = pd.DataFrame(all_data)
    
    # Convert dates to string format for JSON serialization (for demo)
    date_cols = ['order_date', 'activation_date', 'churn_date', 'week_ending']
    for col in date_cols:
        if col in df.columns:
            df[col] = df[col].dt.strftime('%Y-%m-%d')
    
    return df, reps

# ==================== AUTHENTICATION ====================
def setup_authentication() -> Dict:
    """Setup token-based authentication system"""
    
    # In production, this would come from secrets or database
    # For demo, we'll create a simple token system
    tokens = {
        "hunter_token": {
            "rep_id": "REP001",
            "rep_name": "Hunter Stockwell",
            "role": "Rep",
            "territory": "North",
            "monthly_quota": 150000,
            "token": "hunter_token"
        },
        "alex_token": {
            "rep_id": "REP002",
            "rep_name": "Alex Johnson",
            "role": "Senior Rep",
            "territory": "South",
            "monthly_quota": 200000,
            "token": "alex_token"
        },
        "demo_token": {
            "rep_id": "REP003",
            "rep_name": "Demo User",
            "role": "Rep",
            "territory": "Central",
            "monthly_quota": 120000,
            "token": "demo_token"
        }
    }
    
    # Store in session state
    if 'tokens' not in st.session_state:
        st.session_state.tokens = tokens
    
    return tokens

def authenticate_user() -> Optional[Dict]:
    """Authenticate user based on token in URL or session"""
    
    tokens = setup_authentication()
    
    # Check for token in query parameters
    query_params = st.query_params
    token_from_url = query_params.get("token", "")
    
    # Check for token in session (if already logged in)
    token_from_session = st.session_state.get('auth_token', '')
    
    # Use token from URL or session
    token = token_from_url or token_from_session
    
    if not token:
        # Show login screen
        show_login_screen(tokens)
        return None
    
    # Validate token
    if token in tokens:
        st.session_state.auth_token = token
        return tokens[token]
    else:
        st.error("Invalid access token. Please use a valid token.")
        show_login_screen(tokens)
        return None

def show_login_screen(tokens: Dict):
    """Display login screen for demo"""
    st.title("🔐 Palmetto Performance Dashboard")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.info("This is a demo dashboard. Please select a rep to continue:")
        
        selected_rep = st.selectbox(
            "Choose a demo user:",
            options=list(tokens.keys()),
            format_func=lambda x: tokens[x]['rep_name']
        )
        
        if st.button("Access Dashboard", type="primary"):
            # Set the token and reload
            st.session_state.auth_token = selected_rep
            st.rerun()
        
        st.markdown("---")
        st.caption("In production, each rep would receive a unique private link.")

# ==================== DATA PROCESSING ====================
def filter_rep_data(df: pd.DataFrame, rep_id: str) -> pd.DataFrame:
    """Filter data for specific rep"""
    return df[df['rep_id'] == rep_id].copy()

def calculate_metrics(df: pd.DataFrame, rep_info: Dict) -> Dict:
    """Calculate all metrics for dashboard"""
    
    # Convert string dates back to datetime for calculations
    date_cols = ['order_date', 'activation_date', 'churn_date', 'week_ending']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Current month calculations
    current_month = datetime.now().replace(day=1)
    month_mask = df['order_date'] >= current_month
    
    # Status counts
    status_counts = df['status'].value_counts().to_dict()
    
    # Order metrics
    total_orders = len(df)
    month_orders = len(df[month_mask])
    
    # Value metrics
    total_value = df['order_value'].sum()
    month_value = df[month_mask]['order_value'].sum()
    
    # Active orders (not cancelled or churned)
    active_orders = len(df[df['status'] == 'Active'])
    
    # Activation metrics
    activated_mask = df['status'].isin(['Active', 'Pending'])
    activated_orders = len(df[activated_mask])
    month_activated = len(df[month_mask & activated_mask])
    
    # Churn metrics
    churned_orders = len(df[df['status'] == 'Churned'])
    
    # Time to activation
    activation_times = []
    for _, row in df.iterrows():
        if row['activation_date'] and pd.notnull(row['activation_date']):
            if pd.notnull(row['order_date']):
                days = (row['activation_date'] - row['order_date']).days
                if days >= 0:
                    activation_times.append(days)
    
    avg_activation_days = np.mean(activation_times) if activation_times else 0
    
    # Progress to quota
    monthly_quota = rep_info.get('monthly_quota', 150000)
    quota_progress = (month_value / monthly_quota) * 100 if monthly_quota > 0 else 0
    
    return {
        'total_orders': total_orders,
        'month_orders': month_orders,
        'total_value': total_value,
        'month_value': month_value,
        'active_orders': active_orders,
        'activated_orders': activated_orders,
        'month_activated': month_activated,
        'churned_orders': churned_orders,
        'status_counts': status_counts,
        'avg_activation_days': avg_activation_days,
        'quota_progress': quota_progress,
        'monthly_quota': monthly_quota
    }

def prepare_chart_data(df: pd.DataFrame) -> Dict:
    """Prepare data for charts"""
    
    # Weekly trends
    df['week_ending'] = pd.to_datetime(df['week_ending'])
    weekly_data = df.groupby(['week_ending', 'status']).size().unstack(fill_value=0).reset_index()
    
    # Monthly trends
    df['month'] = df['order_date'].dt.strftime('%Y-%m')
    monthly_data = df.groupby(['month', 'status']).size().unstack(fill_value=0).reset_index()
    
    # Status distribution over time
    status_over_time = df.groupby(['week_ending']).agg({
        'order_value': 'sum',
        'order_id': 'count'
    }).reset_index()
    
    return {
        'weekly': weekly_data,
        'monthly': monthly_data,
        'status_over_time': status_over_time
    }

# ==================== CHART FUNCTIONS ====================
def create_weekly_activations_chart(weekly_data: pd.DataFrame) -> go.Figure:
    """Create weekly activations chart"""
    
    fig = go.Figure()
    
    # Only show last 12 weeks for clarity
    if len(weekly_data) > 12:
        weekly_data = weekly_data.tail(12)
    
    # Colors for statuses
    status_colors = {
        'Active': '#2E8B57',
        'Pending': '#FFA500',
        'Cancelled': '#DC143C',
        'Churned': '#A9A9A9',
        'No Installation Scheduled': '#4682B4'
    }
    
    for status in ['Active', 'Pending', 'Cancelled']:
        if status in weekly_data.columns:
            fig.add_trace(go.Scatter(
                x=weekly_data['week_ending'],
                y=weekly_data[status],
                mode='lines+markers',
                name=status,
                line=dict(color=status_colors.get(status, '#000000'), width=2),
                marker=dict(size=8)
            ))
    
    fig.update_layout(
        title="Weekly Activations by Status",
        xaxis_title="Week Ending Date",
        yaxis_title="Number of Orders",
        hovermode='x unified',
        height=400,
        plot_bgcolor='rgba(240, 240, 240, 0.1)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        xaxis=dict(
            gridcolor='rgba(200, 200, 200, 0.2)',
            tickformat="%b %d"
        ),
        yaxis=dict(
            gridcolor='rgba(200, 200, 200, 0.2)'
        )
    )
    
    return fig

def create_value_trend_chart(status_over_time: pd.DataFrame) -> go.Figure:
    """Create order value trend chart"""
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=status_over_time['week_ending'],
        y=status_over_time['order_value'].rolling(window=4, min_periods=1).mean(),  # 4-week moving average
        mode='lines',
        name='Order Value (4-week avg)',
        line=dict(color='#1E90FF', width=3),
        fill='tozeroy',
        fillcolor='rgba(30, 144, 255, 0.1)'
    ))
    
    fig.update_layout(
        title="Order Value Trend (Smoothed)",
        xaxis_title="Week Ending",
        yaxis_title="Order Value ($)",
        height=350,
        plot_bgcolor='rgba(240, 240, 240, 0.1)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        showlegend=True
    )
    
    return fig

def create_status_pie_chart(status_counts: Dict) -> go.Figure:
    """Create pie chart for status distribution"""
    
    labels = list(status_counts.keys())
    values = list(status_counts.values())
    
    # Colors
    colors = ['#2E8B57', '#FFA500', '#DC143C', '#A9A9A9', '#4682B4']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker=dict(colors=colors[:len(labels)]),
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Status Distribution",
        height=350,
        showlegend=False
    )
    
    return fig

# ==================== DASHBOARD COMPONENTS ====================
def render_header(rep_info: Dict):
    """Render dashboard header"""
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        st.title("📊 Palmetto Performance Dashboard")
        st.markdown(f"**Your Performance** • Representative: **{rep_info['rep_name']}**")
    
    with col2:
        st.metric(
            "Today",
            datetime.now().strftime("%b %d, %Y"),
            help="Current date"
        )
    
    with col3:
        st.metric(
            "Role",
            rep_info['role'],
            help="Your current role"
        )
    
    with col4:
        # Current streak (demo data)
        streak = random.randint(1, 30)
        st.metric(
            "Current Streak",
            f"{streak} days",
            help="Consecutive days meeting activity goals"
        )
    
    st.divider()

def render_metrics_section(metrics: Dict):
    """Render main metrics section"""
    
    st.header("Overall Summary")
    
    # Row 1: Order counts
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Orders",
            f"{metrics['total_orders']:,}",
            help="All-time total orders placed"
        )
    
    with col2:
        st.metric(
            "MTD Orders",
            f"{metrics['month_orders']:,}",
            help="Month-to-date orders"
        )
    
    with col3:
        st.metric(
            "Total Value",
            f"${metrics['total_value']:,.0f}",
            help="All-time order value"
        )
    
    with col4:
        st.metric(
            "MTD Value",
            f"${metrics['month_value']:,.0f}",
            help="Month-to-date order value"
        )
    
    with col5:
        delta = metrics['quota_progress'] - 100 if metrics['quota_progress'] > 100 else metrics['quota_progress']
        st.metric(
            "Quota Progress",
            f"{metrics['quota_progress']:.1f}%",
            delta=f"{delta:.1f}%",
            help=f"Progress toward ${metrics['monthly_quota']:,.0f} monthly quota"
        )
    
    st.divider()
    
    # Row 2: Status breakdown
    st.subheader("Status Breakdown")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    status_mapping = {
        'Active': metrics['status_counts'].get('Active', 0),
        'Pending': metrics['status_counts'].get('Pending', 0),
        'Cancelled': metrics['status_counts'].get('Cancelled', 0),
        'Churned': metrics['status_counts'].get('Churned', 0),
        'No Installation Scheduled': metrics['status_counts'].get('No Installation Scheduled', 0)
    }
    
    # Define colors for each status
    status_colors = {
        'Active': '#2E8B57',
        'Pending': '#FFA500',
        'Cancelled': '#DC143C',
        'Churned': '#A9A9A9',
        'No Installation Scheduled': '#4682B4'
    }
    
    for i, (status, count) in enumerate(status_mapping.items()):
        with [col1, col2, col3, col4, col5][i]:
            # Create a custom HTML card for better styling
            card_html = f"""
            <div style="
                background-color: {status_colors[status]}20;
                border-left: 4px solid {status_colors[status]};
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
            ">
                <h3 style="margin: 0; color: #333; font-size: 1.5rem; font-weight: bold;">
                    {count}
                </h3>
                <p style="margin: 0.25rem 0 0 0; color: #666; font-size: 0.9rem;">
                    {status}
                </p>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
    
    st.divider()

def render_charts_section(chart_data: Dict, metrics: Dict):
    """Render charts section"""
    
    st.header("Performance Trends")
    
    # Row 1: Charts
    col1, col2 = st.columns(2)
    
    with col1:
        weekly_chart = create_weekly_activations_chart(chart_data['weekly'])
        st.plotly_chart(weekly_chart, use_container_width=True)
    
    with col2:
        value_chart = create_value_trend_chart(chart_data['status_over_time'])
        st.plotly_chart(value_chart, use_container_width=True)
    
    # Row 2: More metrics and pie chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Activation Metrics")
        
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric(
                "Activated Orders",
                f"{metrics['activated_orders']:,}",
                help="Total orders activated"
            )
        
        with metric_col2:
            st.metric(
                "MTD Activated",
                f"{metrics['month_activated']:,}",
                help="Month-to-date activated orders"
            )
        
        with metric_col3:
            st.metric(
                "Avg Days to Activate",
                f"{metrics['avg_activation_days']:.1f}",
                help="Average days from order to activation"
            )
        
        # Order timeline table (recent orders)
        st.subheader("Recent Orders")
        
        # Create sample recent orders table
        recent_orders = pd.DataFrame({
            'Order ID': [f"ORD{1000 + i}" for i in range(5)],
            'Date': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(5)],
            'Value': [f"${random.randint(1000, 5000):,}" for _ in range(5)],
            'Status': random.choices(['Active', 'Pending', 'Cancelled'], k=5)
        })
        
        st.dataframe(
            recent_orders,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Order ID": st.column_config.TextColumn("Order ID", width="medium"),
                "Date": st.column_config.TextColumn("Date", width="small"),
                "Value": st.column_config.TextColumn("Value", width="small"),
                "Status": st.column_config.TextColumn("Status", width="small")
            }
        )
    
    with col2:
        # Status distribution pie chart
        pie_chart = create_status_pie_chart(metrics['status_counts'])
        st.plotly_chart(pie_chart, use_container_width=True)
        
        # Performance indicators
        st.subheader("Performance Indicators")
        
        # Quota progress bar
        st.progress(
            min(metrics['quota_progress'] / 100, 1.0),
            text=f"Monthly Quota: ${metrics['month_value']:,.0f} / ${metrics['monthly_quota']:,.0f} ({metrics['quota_progress']:.1f}%)"
        )
        
        # Churn rate
        churn_rate = (metrics['churned_orders'] / max(metrics['activated_orders'], 1)) * 100
        st.metric(
            "Churn Rate",
            f"{churn_rate:.1f}%",
            delta=f"-{random.uniform(0.5, 2.0):.1f}%" if churn_rate < 10 else f"+{random.uniform(0.5, 2.0):.1f}%",
            help="Percentage of activated orders that churned"
        )
        
        # Territory ranking (demo)
        st.metric(
            "Territory Rank",
            f"#{random.randint(1, 5)}",
            help="Your rank within your territory"
        )
    
    st.divider()

def render_footer():
    """Render dashboard footer"""
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        st.caption("📊 Palmetto Performance Dashboard • v1.0 • Data refreshes daily")
        st.caption("For support, contact your manager or IT department")
        
        # Demo controls
        with st.expander("Demo Controls"):
            st.info("This is a demo dashboard with simulated data.")
            if st.button("Refresh Demo Data"):
                st.cache_data.clear()
                st.rerun()
            
            st.code("""
            # Access different demo users:
            # Hunter: ?token=hunter_token
            # Alex: ?token=alex_token
            # Demo: ?token=demo_token
            """, language="markdown")

# ==================== MAIN APP ====================
def main():
    """Main application function"""
    
    # Initialize session state
    if 'data' not in st.session_state:
        with st.spinner("Generating test data..."):
            st.session_state.data, st.session_state.reps = generate_test_data()
    
    # Authenticate user
    rep_info = authenticate_user()
    
    if not rep_info:
        return  # Stop if not authenticated
    
    # Get data for this rep
    df = filter_rep_data(st.session_state.data, rep_info['rep_id'])
    
    if df.empty:
        st.warning(f"No data found for {rep_info['rep_name']}. Using sample data...")
        # Generate sample data for this rep if none exists
        df, _ = generate_test_data(num_reps=1, num_days=30)
    
    # Calculate metrics
    metrics = calculate_metrics(df, rep_info)
    
    # Prepare chart data
    chart_data = prepare_chart_data(df)
    
    # Render dashboard
    render_header(rep_info)
    render_metrics_section(metrics)
    render_charts_section(chart_data, metrics)
    render_footer()

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()