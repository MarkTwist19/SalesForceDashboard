# app_simple.py - Ultra-simple working version
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import random

# Page configuration
st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Sales Performance Dashboard")

# Sidebar for user selection
st.sidebar.title("User Login")
user = st.sidebar.selectbox(
    "Select Demo User:",
    ["Hunter Stockwell (Rep)", "Alex Johnson (Senior Rep)", "Taylor Smith (Rep)", "Morgan Williams (Manager)"]
)

# Extract rep info
rep_info = {
    "Hunter Stockwell (Rep)": {
        "name": "Hunter Stockwell",
        "role": "Rep",
        "quota": 150000,
        "id": "REP001"
    },
    "Alex Johnson (Senior Rep)": {
        "name": "Alex Johnson",
        "role": "Senior Rep",
        "quota": 200000,
        "id": "REP002"
    },
    "Taylor Smith (Rep)": {
        "name": "Taylor Smith",
        "role": "Rep",
        "quota": 120000,
        "id": "REP003"
    },
    "Morgan Williams (Manager)": {
        "name": "Morgan Williams",
        "role": "Manager",
        "quota": 500000,
        "id": "MGR001"
    }
}

current_user = rep_info[user]

# Header
st.markdown(f"### Welcome, {current_user['name']}")
st.markdown(f"**Role:** {current_user['role']} | **Today:** {datetime.now().strftime('%B %d, %Y')}")
st.divider()

# Generate demo data
@st.cache_data
def generate_demo_data(rep_id):
    np.random.seed(hash(rep_id) % (2**32))
    
    data = []
    for i in range(100):
        order_date = datetime.now() - timedelta(days=random.randint(0, 180))
        status = random.choices(
            ['Active', 'Pending', 'Cancelled', 'Churned', 'No Installation Scheduled'],
            weights=[0.6, 0.2, 0.1, 0.07, 0.03]
        )[0]
        
        data.append({
            'order_id': f"ORD{1000 + i}",
            'order_date': order_date,
            'value': random.randint(1000, 15000),
            'status': status,
            'product': random.choice(['Solar', 'Battery', 'EV Charger', 'Full System'])
        })
    
    return pd.DataFrame(data)

# Load data
df = generate_demo_data(current_user['id'])

# Calculate metrics
total_orders = len(df)
active_orders = len(df[df['status'] == 'Active'])
pending_orders = len(df[df['status'] == 'Pending'])
cancelled_orders = len(df[df['status'] == 'Cancelled'])
churned_orders = len(df[df['status'] == 'Churned'])
total_value = df['value'].sum()

# MTD calculations
current_month = datetime.now().replace(day=1)
mtd_df = df[df['order_date'] >= current_month]
mtd_orders = len(mtd_df)
mtd_value = mtd_df['value'].sum()

# Progress to quota
quota_progress = (mtd_value / current_user['quota']) * 100

# Display metrics
st.header("📈 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Orders", f"{total_orders:,}")
    st.metric("MTD Orders", f"{mtd_orders:,}")

with col2:
    st.metric("Total Value", f"${total_value:,.0f}")
    st.metric("MTD Value", f"${mtd_value:,.0f}")

with col3:
    st.metric("Active Orders", f"{active_orders:,}")
    st.metric("Pending Orders", f"{pending_orders:,}")

with col4:
    st.metric("Quota Progress", f"{quota_progress:.1f}%")
    st.progress(min(quota_progress / 100, 1.0))

st.divider()

# Status breakdown
st.header("📊 Status Breakdown")

status_cols = st.columns(5)

statuses = ['Active', 'Pending', 'Cancelled', 'Churned', 'No Installation Scheduled']
counts = [active_orders, pending_orders, cancelled_orders, churned_orders, 
          len(df[df['status'] == 'No Installation Scheduled'])]
colors = ['#2E8B57', '#FFA500', '#DC143C', '#A9A9A9', '#4682B4']

for i, (status, count, color) in enumerate(zip(statuses, counts, colors)):
    with status_cols[i]:
        st.markdown(f"""
        <div style='background-color:{color}20; padding:20px; border-radius:10px; text-align:center; border-left:5px solid {color}'>
            <h3 style='margin:0;'>{count}</h3>
            <p style='margin:0; color:#666;'>{status}</p>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# Charts
st.header("📈 Performance Charts")

# Weekly trend
df['week'] = df['order_date'].dt.strftime('%Y-%U')
weekly_data = df.groupby('week').agg({'value': 'sum', 'order_id': 'count'}).reset_index()

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=weekly_data['week'],
    y=weekly_data['value'],
    mode='lines+markers',
    name='Weekly Sales',
    line=dict(color='#1E90FF', width=3)
))

fig1.update_layout(
    title="Weekly Sales Trend",
    xaxis_title="Week",
    yaxis_title="Sales ($)",
    height=350
)

st.plotly_chart(fig1, use_container_width=True)

# Status distribution
status_counts = df['status'].value_counts()

fig2 = go.Figure(data=[go.Pie(
    labels=status_counts.index,
    values=status_counts.values,
    hole=.3,
    marker=dict(colors=colors)
)])

fig2.update_layout(
    title="Order Status Distribution",
    height=350
)

st.plotly_chart(fig2, use_container_width=True)

# Recent orders
st.header("📋 Recent Orders")
st.dataframe(
    df.sort_values('order_date', ascending=False).head(10)[['order_id', 'order_date', 'value', 'status', 'product']],
    hide_index=True
)

# Footer
st.divider()
st.caption(f"Demo Dashboard • User: {current_user['name']} • Data refreshes on reload")
st.caption("For demonstration purposes only")