# app.py - Fixed working prototype (Backward Compatible)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import random
from typing import Dict, Optional
import urllib.parse

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="Palmetto Performance Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== HELPER FUNCTIONS ====================
def get_query_params():
    """Get query parameters - compatible with older Streamlit versions"""
    try:
        # Try the new way first (Streamlit >= 1.28.0)
        return st.query_params
    except AttributeError:
        try:
            # Try the experimental way (older versions)
            return st.experimental_get_query_params()
        except AttributeError:
            # Fall back to manual parsing
            return {}

# ==================== TEST DATA GENERATION ====================
def generate_test_data(rep_id: str, rep_name: str) -> pd.DataFrame:
    """Generate realistic test sales data for a specific rep"""
    
    # Set random seed for reproducibility based on rep_id
    seed = abs(hash(rep_id)) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    
    statuses = ['Active', 'Pending', 'Cancelled', 'Churned', 'No Installation Scheduled']
    products = ['Solar Panel', 'Battery Storage', 'EV Charger', 'Full System']
    territories = ['North', 'South', 'East', 'West']
    
    # Generate 150-300 orders per rep
    num_orders = random.randint(150, 300)
    all_data = []
    
    # Start generating from 6 months ago
    start_date = datetime.now() - timedelta(days=180)
    
    for i in range(num_orders):
        order_id = f"ORD{1000 + i}"
        
        # Order date (past 180 days)
        days_ago = random.randint(0, 180)
        order_date = datetime.now() - timedelta(days=days_ago)
        
        # Order value
        order_value = round(random.uniform(1000, 15000), 2)
        
        # Status with realistic distribution
        status_weights = [0.6, 0.2, 0.1, 0.07, 0.03]
        status = random.choices(statuses, weights=status_weights)[0]
        
        # Activation date
        activation_date = None
        if status in ['Active', 'Pending']:
            activation_delay = random.randint(1, 30)
            activation_date = order_date + timedelta(days=activation_delay)
        
        # Churn date
        churn_date = None
        if status == 'Churned' and activation_date:
            churn_delay = random.randint(30, 120)
            churn_date = activation_date + timedelta(days=churn_delay)
        
        # Week ending (Saturday)
        days_to_saturday = (5 - order_date.weekday()) % 7
        week_ending = order_date + timedelta(days=days_to_saturday)
        
        order_data = {
            'rep_id': rep_id,
            'rep_name': rep_name,
            'order_id': order_id,
            'order_date': order_date,
            'order_value': order_value,
            'status': status,
            'activation_date': activation_date,
            'churn_date': churn_date,
            'week_ending': week_ending,
            'territory': random.choice(territories),
            'product_type': random.choice(products),
            'sales_channel': random.choice(['Direct', 'Referral', 'Partner']),
            'customer_type': random.choice(['Residential', 'Commercial'])
        }
        
        all_data.append(order_data)
    
    return pd.DataFrame(all_data)

# ==================== AUTHENTICATION ====================
def setup_authentication() -> Dict:
    """Setup token-based authentication system"""
    
    tokens = {
        "hunter_token": {
            "rep_id": "REP001",
            "rep_name": "Hunter Stockwell",
            "role": "Rep",
            "territory": "North",
            "monthly_quota": 150000,
            "team": "Alpha",
            "hire_date": "2023-01-15"
        },
        "alex_token": {
            "rep_id": "REP002",
            "rep_name": "Alex Johnson",
            "role": "Senior Rep",
            "territory": "South",
            "monthly_quota": 200000,
            "team": "Beta",
            "hire_date": "2022-06-10"
        },
        "demo_token": {
            "rep_id": "REP003",
            "rep_name": "Taylor Smith",
            "role": "Rep",
            "territory": "Central",
            "monthly_quota": 120000,
            "team": "Gamma",
            "hire_date": "2024-01-20"
        },
        "manager_token": {
            "rep_id": "MGR001",
            "rep_name": "Morgan Williams",
            "role": "Sales Manager",
            "territory": "All",
            "monthly_quota": 500000,
            "team": "All Teams",
            "hire_date": "2021-03-01"
        }
    }
    
    return tokens

def get_token_from_url():
    """Extract token from URL - works with all Streamlit versions"""
    query_params = get_query_params()
    
    # Handle different return types
    if isinstance(query_params, dict):
        if 'token' in query_params:
            token_value = query_params['token']
            # Handle both list and single value
            if isinstance(token_value, list):
                return token_value[0] if token_value else ""
            return token_value
    return ""

def authenticate_user() -> Optional[Dict]:
    """Authenticate user based on token"""
    
    tokens = setup_authentication()
    
    # Get token from URL
    token_from_url = get_token_from_url()
    
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
        st.error("❌ Invalid access token.")
        show_login_screen(tokens)
        return None

def show_login_screen(tokens: Dict):
    """Display login screen for demo"""
    
    st.markdown("<h1 style='text-align: center;'>🔐 Palmetto Performance Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.info("""
    ### Welcome to the Sales Performance Dashboard
    
    This is a demo dashboard with simulated sales data. Select a user below to explore the dashboard.
    """)
    
    # User selection
    st.subheader("👥 Select a Demo User")
    
    cols = st.columns(2)
    
    for idx, (token, user_info) in enumerate(tokens.items()):
        col = cols[idx % 2]
        
        with col:
            with st.container():
                st.markdown(f"""
                <div style='border: 1px solid #ddd; border-radius: 10px; padding: 20px; margin: 10px 0; background-color: #f9f9f9;'>
                    <h3 style='margin-top: 0;'>{user_info['rep_name']}</h3>
                    <p><strong>Role:</strong> {user_info['role']}</p>
                    <p><strong>Territory:</strong> {user_info['territory']}</p>
                    <p><strong>Monthly Quota:</strong> ${user_info['monthly_quota']:,}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Login as {user_info['rep_name']}", key=f"login_{token}"):
                    st.session_state.auth_token = token
                    st.rerun()
    
    st.markdown("---")
    st.caption("💡 **Tip:** In production, each rep receives a unique private link")
    st.caption("🔗 Example URL: https://dashboard.streamlit.app/?token=unique_token_here")

# ==================== DATA PROCESSING ====================
def calculate_metrics(df: pd.DataFrame, rep_info: Dict) -> Dict:
    """Calculate all metrics for dashboard"""
    
    # Ensure date columns are datetime
    date_columns = ['order_date', 'activation_date', 'churn_date', 'week_ending']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Current month calculations
    current_month = datetime.now().replace(day=1)
    current_month_str = current_month.strftime('%Y-%m')
    
    # Create month mask
    month_mask = pd.Series(False, index=df.index)
    if 'order_date' in df.columns and not df['order_date'].isna().all():
        month_mask = df['order_date'].dt.strftime('%Y-%m') == current_month_str
    
    # Status counts
    status_counts = {}
    for status in ['Active', 'Pending', 'Cancelled', 'Churned', 'No Installation Scheduled']:
        count = len(df[df['status'] == status])
        status_counts[status] = count
    
    # Order metrics
    total_orders = len(df)
    month_orders = len(df[month_mask]) if month_mask.any() else 0
    
    # Value metrics
    total_value = df['order_value'].sum()
    month_value = df.loc[month_mask, 'order_value'].sum() if month_mask.any() else 0
    
    # Active orders
    active_orders = status_counts.get('Active', 0)
    
    # Activated orders (Active + Pending)
    activated_orders = status_counts.get('Active', 0) + status_counts.get('Pending', 0)
    
    # Month activated
    month_activated = 0
    if month_mask.any():
        month_activated_mask = month_mask & df['status'].isin(['Active', 'Pending'])
        month_activated = len(df[month_activated_mask])
    
    # Churn rate
    churned_orders = status_counts.get('Churned', 0)
    churn_rate = (churned_orders / max(activated_orders, 1)) * 100
    
    # Time to activation
    activation_times = []
    for _, row in df.iterrows():
        if pd.notna(row.get('activation_date')) and pd.notna(row.get('order_date')):
            days = (row['activation_date'] - row['order_date']).days
            if days >= 0:
                activation_times.append(days)
    
    avg_activation_days = np.mean(activation_times) if activation_times else 0
    
    # Progress to quota
    monthly_quota = rep_info.get('monthly_quota', 150000)
    quota_progress = (month_value / monthly_quota) * 100 if monthly_quota > 0 else 0
    
    # Streak calculation (demo)
    streak = random.randint(1, 30)
    
    # Territory ranking (demo)
    territory_rank = random.randint(1, 5)
    
    # Average order value
    avg_order_value = total_value / max(total_orders, 1)
    
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
        'avg_activation_days': round(avg_activation_days, 1),
        'quota_progress': round(quota_progress, 1),
        'monthly_quota': monthly_quota,
        'churn_rate': round(churn_rate, 1),
        'streak': streak,
        'territory_rank': territory_rank,
        'avg_order_value': round(avg_order_value, 0),
        'current_month': current_month.strftime('%B %Y')
    }

def prepare_chart_data(df: pd.DataFrame) -> Dict:
    """Prepare data for charts"""
    
    # Make a copy to avoid modifying original
    df_copy = df.copy()
    
    # Ensure week_ending is datetime
    if 'week_ending' in df_copy.columns:
        df_copy['week_ending'] = pd.to_datetime(df_copy['week_ending'], errors='coerce')
    
    # Weekly trends (last 12 weeks)
    weekly_data = pd.DataFrame()
    if 'week_ending' in df_copy.columns and 'status' in df_copy.columns:
        # Filter for last 12 weeks
        twelve_weeks_ago = datetime.now() - timedelta(weeks=12)
        df_recent = df_copy[df_copy['week_ending'] >= twelve_weeks_ago].copy()
        
        if not df_recent.empty:
            # Group by week and status
            weekly_data = df_recent.groupby(['week_ending', 'status']).size().unstack(fill_value=0).reset_index()
    
    # Status distribution over time
    status_over_time = pd.DataFrame()
    if 'week_ending' in df_copy.columns and 'order_value' in df_copy.columns:
        status_over_time = df_copy.groupby('week_ending').agg({
            'order_value': 'sum',
            'order_id': 'count'
        }).reset_index()
    
    return {
        'weekly': weekly_data,
        'status_over_time': status_over_time
    }

# ==================== CHART FUNCTIONS ====================
def create_weekly_activations_chart(weekly_data: pd.DataFrame) -> go.Figure:
    """Create weekly activations chart"""
    
    fig = go.Figure()
    
    if weekly_data.empty or len(weekly_data) < 2:
        # Return empty chart with message
        fig.add_annotation(
            text="No weekly data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig
    
    # Sort by date
    weekly_data = weekly_data.sort_values('week_ending')
    
    # Colors for statuses
    status_colors = {
        'Active': '#2E8B57',
        'Pending': '#FFA500',
        'Cancelled': '#DC143C',
        'Churned': '#A9A9A9',
        'No Installation Scheduled': '#4682B4'
    }
    
    # Plot each status
    for status in ['Active', 'Pending', 'Cancelled']:
        if status in weekly_data.columns:
            fig.add_trace(go.Scatter(
                x=weekly_data['week_ending'],
                y=weekly_data[status],
                mode='lines+markers',
                name=status,
                line=dict(color=status_colors.get(status, '#000000'), width=2),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title="Weekly Activations by Status",
        xaxis_title="Week Ending",
        yaxis_title="Number of Orders",
        hovermode='x unified',
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            gridcolor='rgba(200, 200, 200, 0.2)',
            tickformat="%b %d",
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(200, 200, 200, 0.2)',
            showgrid=True
        )
    )
    
    return fig

def create_value_trend_chart(status_over_time: pd.DataFrame) -> go.Figure:
    """Create order value trend chart"""
    
    fig = go.Figure()
    
    if status_over_time.empty or len(status_over_time) < 2:
        # Return empty chart with message
        fig.add_annotation(
            text="No trend data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig
    
    # Sort by date
    status_over_time = status_over_time.sort_values('week_ending')
    
    # Calculate 4-week moving average
    status_over_time['order_value_ma'] = status_over_time['order_value'].rolling(
        window=4, min_periods=1
    ).mean()
    
    fig.add_trace(go.Scatter(
        x=status_over_time['week_ending'],
        y=status_over_time['order_value_ma'],
        mode='lines',
        name='Order Value (4-week avg)',
        line=dict(color='#1E90FF', width=3),
        fill='tozeroy',
        fillcolor='rgba(30, 144, 255, 0.1)'
    ))
    
    fig.update_layout(
        title="Order Value Trend",
        xaxis_title="Week Ending",
        yaxis_title="Order Value ($)",
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        yaxis=dict(
            tickprefix="$",
            tickformat=","
        )
    )
    
    return fig

def create_status_pie_chart(status_counts: Dict) -> go.Figure:
    """Create pie chart for status distribution"""
    
    # Filter out zero values
    filtered_counts = {k: v for k, v in status_counts.items() if v > 0}
    
    if not filtered_counts:
        fig = go.Figure()
        fig.add_annotation(
            text="No status data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig
    
    labels = list(filtered_counts.keys())
    values = list(filtered_counts.values())
    
    # Colors
    colors = ['#2E8B57', '#FFA500', '#DC143C', '#A9A9A9', '#4682B4']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker=dict(colors=colors[:len(labels)]),
        textinfo='label+percent',
        textposition='outside',
        hoverinfo='label+value+percent'
    )])
    
    fig.update_layout(
        title="Status Distribution",
        height=350,
        showlegend=False,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

# ==================== DASHBOARD COMPONENTS ====================
def render_header(rep_info: Dict, metrics: Dict):
    """Render dashboard header"""
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.title("📊 Palmetto Performance Dashboard")
        st.markdown(f"**Your Performance** • **{rep_info['rep_name']}** • {rep_info['role']}")
    
    with col2:
        st.markdown("### Current Status")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Today", datetime.now().strftime("%b %d, %Y"))
        with cols[1]:
            st.metric("Role", rep_info['role'])
        with cols[2]:
            st.metric("Streak", f"{metrics['streak']} days")
    
    st.markdown("---")

def render_key_metrics(metrics: Dict):
    """Render key metrics section"""
    
    st.subheader("📈 Key Performance Indicators")
    
    # Row 1: Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Orders",
            f"{metrics['total_orders']:,}",
            help="All-time total orders placed"
        )
        st.metric(
            "MTD Orders",
            f"{metrics['month_orders']:,}",
            delta=f"{metrics['month_orders'] - random.randint(5, 15)} from last month",
            help=f"Orders in {metrics['current_month']}"
        )
    
    with col2:
        st.metric(
            "Total Value",
            f"${metrics['total_value']:,.0f}",
            help="All-time sales value"
        )
        st.metric(
            "MTD Value",
            f"${metrics['month_value']:,.0f}",
            delta=f"${metrics['month_value'] - random.randint(5000, 15000):,.0f} from last month",
            help=f"Sales value in {metrics['current_month']}"
        )
    
    with col3:
        st.metric(
            "Active Orders",
            f"{metrics['active_orders']:,}",
            help="Currently active orders"
        )
        st.metric(
            "Avg Order Value",
            f"${metrics['avg_order_value']:,.0f}",
            help="Average value per order"
        )
    
    with col4:
        # Quota progress with color coding
        progress_value = min(metrics['quota_progress'] / 100, 1.0)
        
        if metrics['quota_progress'] >= 100:
            progress_color = "green"
        elif metrics['quota_progress'] >= 75:
            progress_color = "orange"
        else:
            progress_color = "red"
        
        st.metric(
            "Quota Progress",
            f"{metrics['quota_progress']}%",
            help=f"Progress toward ${metrics['monthly_quota']:,.0f} monthly quota"
        )
        
        # Progress bar
        st.progress(progress_value, text=f"${metrics['month_value']:,.0f} / ${metrics['monthly_quota']:,.0f}")
    
    st.markdown("---")

def render_status_breakdown(metrics: Dict):
    """Render status breakdown section"""
    
    st.subheader("📊 Order Status Breakdown")
    
    # Create metrics for each status
    cols = st.columns(5)
    
    status_config = {
        'Active': {'color': '#2E8B57', 'icon': '✅', 'desc': 'Active installations'},
        'Pending': {'color': '#FFA500', 'icon': '⏳', 'desc': 'Awaiting activation'},
        'Cancelled': {'color': '#DC143C', 'icon': '❌', 'desc': 'Cancelled orders'},
        'Churned': {'color': '#A9A9A9', 'icon': '↩️', 'desc': 'Lost customers'},
        'No Installation Scheduled': {'color': '#4682B4', 'icon': '📅', 'desc': 'No install date'}
    }
    
    for i, (status, config) in enumerate(status_config.items()):
        count = metrics['status_counts'].get(status, 0)
        
        with cols[i]:
            # Use HTML for better card styling
            card_html = f"""
            <div style='
                background: linear-gradient(135deg, {config['color']}15, {config['color']}05);
                border: 1px solid {config['color']}30;
                border-radius: 10px;
                padding: 15px;
                text-align: center;
                margin: 5px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            '>
                <div style='font-size: 24px; color: {config['color']}; margin-bottom: 5px;'>
                    {config['icon']}
                </div>
                <div style='font-size: 28px; font-weight: bold; color: #333;'>
                    {count}
                </div>
                <div style='font-size: 14px; color: #666; margin-top: 5px; font-weight: 500;'>
                    {status}
                </div>
                <div style='font-size: 12px; color: #888; margin-top: 3px;'>
                    {config['desc']}
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
    
    st.markdown("---")

def render_charts_section(chart_data: Dict, metrics: Dict):
    """Render charts section"""
    
    st.subheader("📈 Performance Trends")
    
    # Create two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        weekly_chart = create_weekly_activations_chart(chart_data['weekly'])
        st.plotly_chart(weekly_chart, use_container_width=True)
    
    with col2:
        value_chart = create_value_trend_chart(chart_data['status_over_time'])
        st.plotly_chart(value_chart, use_container_width=True)
    
    # Additional metrics row
    st.subheader("📋 Detailed Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Activation Rate",
            f"{(metrics['activated_orders'] / max(metrics['total_orders'], 1) * 100):.1f}%",
            help="Percentage of orders activated"
        )
        st.metric(
            "Avg Days to Activate",
            f"{metrics['avg_activation_days']} days",
            help="Average days from order to activation"
        )
    
    with col2:
        st.metric(
            "Churn Rate",
            f"{metrics['churn_rate']}%",
            delta=f"-{random.uniform(0.1, 1.5):.1f}%" if metrics['churn_rate'] < 10 else f"+{random.uniform(0.1, 1.5):.1f}%",
            help="Percentage of activated orders that churned"
        )
        st.metric(
            "Customer Satisfaction",
            f"{random.randint(85, 99)}%",
            help="Customer satisfaction score"
        )
    
    with col3:
        st.metric(
            "Territory Rank",
            f"#{metrics['territory_rank']}",
            help="Your rank within your territory"
        )
        st.metric(
            "On-Time Activation",
            f"{random.randint(88, 98)}%",
            help="Percentage of orders activated on time"
        )
    
    st.markdown("---")

def render_recent_orders(df: pd.DataFrame):
    """Render recent orders table"""
    
    if df.empty:
        return
    
    st.subheader("📋 Recent Orders (Last 10)")
    
    # Get recent orders
    if 'order_date' in df.columns:
        recent_df = df.sort_values('order_date', ascending=False).head(10).copy()
        
        # Format for display
        display_df = recent_df[['order_id', 'order_date', 'order_value', 'status', 'product_type']].copy()
        display_df['order_date'] = display_df['order_date'].dt.strftime('%Y-%m-%d')
        display_df['order_value'] = display_df['order_value'].apply(lambda x: f"${x:,.0f}")
        
        # Display as table
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "order_id": "Order ID",
                "order_date": "Date",
                "order_value": "Value",
                "status": "Status",
                "product_type": "Product"
            }
        )
    
    st.markdown("---")

def render_footer(rep_info: Dict):
    """Render dashboard footer"""
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.caption(f"👤 Logged in as: {rep_info['rep_name']} | {rep_info['role']} | {rep_info['territory']}")
        st.caption("📊 Palmetto Performance Dashboard v1.0 | Data refreshes daily")
        st.caption("📍 For support, contact your manager or IT department")
    
    with col2:
        # Refresh button
        if st.button("🔄 Refresh Data", type="secondary"):
            # Clear cached data
            if 'data' in st.session_state:
                del st.session_state.data
            st.rerun()
    
    # Demo info
    with st.expander("ℹ️ Demo Information"):
        st.info("This is a working prototype with simulated data for demonstration purposes.")
        
        st.write("**Available Demo Users:**")
        st.code("""
        Hunter Stockwell (Rep)
        Alex Johnson (Senior Rep)
        Taylor Smith (Rep)
        Morgan Williams (Manager)
        """)
        
        st.write("**To switch users:** Clear your browser cache or use incognito mode.")

# ==================== MAIN APP ====================
def main():
    """Main application function"""
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
    
    # Authenticate user
    rep_info = authenticate_user()
    
    if not rep_info:
        return  # Stop if not authenticated
    
    # Generate or load data for this rep
    data_key = f"data_{rep_info['rep_id']}"
    
    if data_key not in st.session_state:
        with st.spinner(f"Loading performance data for {rep_info['rep_name']}..."):
            df = generate_test_data(rep_info['rep_id'], rep_info['rep_name'])
            st.session_state[data_key] = df
    
    df = st.session_state[data_key]
    
    # Calculate metrics
    metrics = calculate_metrics(df, rep_info)
    
    # Prepare chart data
    chart_data = prepare_chart_data(df)
    
    # Render dashboard
    render_header(rep_info, metrics)
    render_key_metrics(metrics)
    render_status_breakdown(metrics)
    render_charts_section(chart_data, metrics)
    render_recent_orders(df)
    render_footer(rep_info)

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()