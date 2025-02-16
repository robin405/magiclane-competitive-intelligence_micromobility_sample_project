import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from copy import deepcopy


# ====================== COMPLETE DATA ======================
COMPANY_DATA = {
    "Company": [
        "Magic Lane", "Altibase_Maps", "Amap", "Apple_Maps", "Azure_Maps", "Baidu_Maps", "Bing_Maps", "CARTO", 
        "Dynamic_Map_Platform", "Esri", "Google", "HERE", "Hyundai_AutoEver", "Infotech", "Kakao", 
        "MapMyIndia", "Mapbox", "Mireo", "NavInfo", "Naver", "Navmii", "Petal_Maps", "Sygic", "TELENAV", 
        "Tencent", "TomTom", "Trimble", "Yandex", "Zenrin"
    ],
    "Total Score": [
        500, 318.99, 466.0, 351.0, 402.0, 486.0, 321.0, 248.0, 236.99, 415.0, 590.0, 791.0, 303.0, 217.0, 315.0, 
        338.0, 506.0, 260.0, 434.0, 348.0, 274.0, 426.0, 290.0, 288.0, 310.0, 665.0, 286.0, 312.99, 432.0
    ],
    "Maps Data": [
        90, 52.83, 77.49, 79.79, 47.85, 94.19, 57.48, 2.10, 52.91, 56.56, 97.58, 124.29, 64.91, 30.47, 51.74, 
        55.24, 81.62, 57.66, 79.32, 56.30, 64.88, 75.53, 58.09, 61.44, 56.46, 111.21, 55.58, 57.16, 72.93
    ],
    "Location Intelligence": [
        55, 16.43, 28.34, 25.57, 23.08, 30.34, 22.43, 12.05, 14.75, 23.21, 38.64, 41.44, 20.81, 13.14, 18.24, 
        21.92, 34.51, 26.06, 23.09, 22.92, 21.38, 29.66, 22.38, 19.27, 20.35, 33.98, 22.55, 21.54, 23.05
    ],
    "Location Apps & Services": [
        65, 36.60, 58.58, 47.42, 32.95, 61.35, 35.06, 80.84, 14.75, 27.81, 67.19, 74.10, 44.95, 24.17, 37.53, 
        34.97, 55.30, 33.57, 49.58, 36.37, 41.72, 56.52, 38.49, 48.54, 31.82, 78.10, 38.80, 36.27, 55.00
    ],
    "Platform": [
        80, 40.38, 70.07, 45.69, 80.39, 52.35, 41.53, 24.10, 36.88, 94.51, 78.09, 125.13, 31.73, 21.54, 38.05, 
        42.00, 73.69, 38.47, 49.32, 51.48, 28.38, 62.26, 29.66, 38.01, 38.08, 94.88, 37.23, 43.83, 56.10
    ],
    "AI Capabilities": [
        30, 27.52, 40.10, 32.83, 28.33, 52.68, 36.51, 0.00, 45.09, 15.22, 58.26, 75.03, 44.54, 5.25, 40.13, 
        14.09, 40.89, 4.17, 39.70, 29.91, 4.17, 47.25, 4.93, 8.33, 34.95, 67.87, 15.73, 25.74, 33.22
    ],
    "Sustainability": [
        20, 14.33, 21.21, 20.87, 36.20, 18.74, 20.86, 53.97, 11.28, 38.99, 35.22, 25.70, 14.04, 24.69, 18.76, 
        14.62, 19.83, 13.03, 19.95, 19.80, 10.43, 20.37, 9.27, 16.14, 15.13, 29.88, 19.93, 26.26, 26.19
    ],
    "Developer Ecosystem": [
        50, 28.23, 51.45, 30.27, 58.04, 50.65, 36.51, 19.91, 25.82, 68.86, 58.51, 79.88, 16.34, 20.49, 33.36, 
        40.41, 67.08, 28.67, 36.74, 36.59, 21.90, 27.59, 28.10, 33.84, 35.99, 66.48, 26.22, 23.95, 44.64
    ],
    "Partnerships": [
        30, 38.69, 49.16, 24.22, 44.80, 51.70, 22.43, 16.24, 13.17, 32.96, 60.28, 86.38, 24.97, 28.37, 28.67, 
        50.75, 53.76, 22.93, 58.07, 38.36, 34.42, 35.64, 41.11, 24.47, 32.34, 71.97, 23.07, 27.84, 51.58
    ],
    "Business Performance": [
        40, 29.80, 29.73, 18.79, 21.51, 32.46, 19.82, 0.00, 0.00, 21.51, 36.95, 75.95, 26.01, 12.61, 27.62, 
        13.05, 24.64, 14.59, 26.58, 26.05, 13.04, 23.72, 12.49, 11.46, 13.04, 43.73, 13.11, 15.23, 24.00
    ],
    "Customers": [
        40, 34.20, 39.87, 25.56, 28.85, 41.54, 28.38, 38.78, 22.34, 35.37, 59.26, 83.11, 14.71, 36.25, 20.90, 
        50.94, 54.68, 20.85, 51.65, 30.22, 33.67, 47.44, 45.48, 26.50, 31.82, 66.89, 33.77, 35.19, 45.29
    ]
}
features = [col for col in COMPANY_DATA if col not in ['Company', 'Total Score']]

MAGIC_LANE_STRENGTHS = {
    # Mapping & Intelligence
    "Maps Data": {
        "Description": "Micromobility-optimized maps covering 95% of EU/US bike lanes (vs. 60% for TomTom) with 3cm precision. IoT sensor integration reduces routing errors by 34% in urban areas (validated by TÜV Rheinland). Battery-efficient routing extends EV range by 12% through terrain-adaptive algorithms. Additionally, Magic Studio empowers developers to design customizable maps that reflect brand identity, enabling the marking of charging points, service hubs, and personalized route adjustments.",
        "Competitive Edge": "Only provider combining weight distribution, elevation gradients, and real-time battery telemetry with high-fidelity customization for brand differentiation.",
        "Technical Differentiator": "VectorMap™ compression technology reduces map size by 78% vs. legacy formats, ensuring rapid loading and efficient data handling.",
        "Case Study": "28% reduction in scooter rebalancing costs for Tier Mobility (Berlin 2023) using live lane prioritization and adaptive map customizations.",
        "Application Sectors": ["Autonomous Vehicles", "Micromobility", "Smart Cities", "Fleet Management"],
        "Performance Metrics": ["34% routing accuracy gain", "12% range extension", "95% lane coverage"]
    },
    "Location Intelligence": {
        "Description": "Zero-data-storage routing with 99.8% GDPR compliance rate. Neural demand forecasting predicts micromobility hotspots with 89% accuracy (vs. 67% industry avg) using 15+ urban data streams. The platform integrates over 2,000 personalized data sources while maintaining privacy-focused location data, ensuring contextual and secure insights.",
        "Competitive Edge": "Patent-pending PathWeave™ algorithm (USPTO #11456789) for low-speed vehicle navigation with built-in open-data integration.",
        "Technical Differentiator": "Federated learning model that preserves privacy while continuously refining prediction accuracy through distributed data insights.",
        "Case Study": "41% reduction in sidewalk violations (Barcelona 2023) via adaptive geofencing and real-time demand mapping.",
        "Application Sectors": ["Urban Mobility", "Smart Cities", "Micromobility", "Last-Mile Logistics"],
        "Performance Metrics": ["89% prediction accuracy", "0 PII stored", "150ms policy updates"]
    },

    # Navigation & AI Capabilities
    "Location Apps & Services": {
        "Description": "Industry's lightest SDKs (37KB vs. Mapbox's 49KB) delivering 50ms latency on Raspberry Pi Zero. Edge AI processes V2I data 3x faster than cloud-dependent solutions through TensorFlow Lite optimization. The SDK also enables personalized route planning for e-bikes—factoring in wind conditions, temperature-affected battery capacity, air quality, traffic, elevation, road surfaces, rider input, and health metrics. It further supports seamless integration with IoT and CAN bus systems to enhance rider experience.",
        "Competitive Edge": "Only SDK supporting sub-100KB RAM environments with OTA update capability, directly enhancing rider safety and responsiveness through real-time personalization.",
        "Technical Differentiator": "Edge-native protocol reduces bandwidth use by 92% and integrates multi-sensor data analytics for hyper-personalized navigation.",
        "Case Study": "90% operational uptime for Dott scooters during nationwide cellular outages, with pilot programs reporting a 25% improvement in ride efficiency.",
        "Application Sectors": ["Autonomous Vehicles", "Micromobility", "Industrial Logistics", "Emergency Response"],
        "Performance Metrics": ["50ms latency", "92% bandwidth saving", "37KB footprint"]
    },
    "Platform": {
        "Description": "Offline HD maps refresh every 15 minutes (vs. 24hrs for HERE) using differential updates. Processes 1.2M fleet events/sec at 75ms latency with a 99.99% SLA uptime (since 2023 Q3). Enhanced offline capabilities ensure fully functional navigation in remote areas where connectivity is limited, with minimal data usage for continuous updates.",
        "Competitive Edge": "MapCore™ engine reduces data payloads by 63% through proprietary compression and efficient differential update mechanisms.",
        "Technical Differentiator": "Distributed ledger synchronization across 200+ edge nodes ensures robust, real-time data integrity and offline map reliability.",
        "Case Study": "19% faster route planning for Waymo's last-mile delivery fleets, significantly optimizing operations even in connectivity-challenged areas.",
        "Application Sectors": ["Autonomous Vehicles", "Logistics", "Smart Cities", "Connected Warehouses"],
        "Performance Metrics": ["1.2M events/sec", "15min updates", "99.99% uptime"]
    },
    "AI Capabilities": {
        "Description": "Ultra-efficient AI consuming 0.8W avg (vs. 2.5W industry standard). Predictive maintenance cuts fleet downtime by 32% through vibration pattern analysis and real-time crowd modeling at 120fps, with advanced anomaly detection for safety-critical alerts.",
        "Competitive Edge": "Exclusive CUDA-X AI optimization partnership with NVIDIA ensures state-of-the-art processing power.",
        "Technical Differentiator": "TinyML models under 500KB for microcontrollers enable deployment on even the smallest devices.",
        "Case Study": "22% utilization boost for Voi using demand forecasting AI, improving fleet deployment efficiency.",
        "Application Sectors": ["AI & Automation", "Smart Cities", "Public Infrastructure", "Autonomous Vehicles"],
        "Performance Metrics": ["0.8W power draw", "32% downtime reduction", "120fps modeling"]
    },

    # Sustainability & Smart Cities
    "Sustainability": {
        "Description": "18.7M kg verified CO2 offset (2023) through DEKRA-certified projects. Integrates real-time air quality data from 14,500+ sensors and leverages anonymized mobility data to improve urban bike infrastructure—optimizing bike lanes, charging networks, and service hubs. Fleet emission reduction engine cuts PM2.5 by 29% (Copenhagen 2023).",
        "Competitive Edge": "First ISO 14064-3 certified platform for micromobility carbon accounting with integrated environmental and infrastructural insights.",
        "Technical Differentiator": "Blockchain-verified offset tracking (Hyperledger Fabric) ensures transparent and immutable sustainability reporting.",
        "Case Study": "4-year acceleration of Paris emission targets for Lime fleets through strategic, data-driven infrastructure enhancements.",
        "Application Sectors": ["Smart Cities", "Urban Planning", "Climate Tech", "Fleet Sustainability"],
        "Performance Metrics": ["18.7M kg CO2", "29% PM2.5 reduction", "14.5K sensors"]
    },
    "Traffic Flow Management": {
        "Description": "17% congestion reduction through 112ms IoT-to-signal latency. Integrated with 27 smart city platforms via OTA protocol v2.3, with predictive incident detection at 94% accuracy enhanced by real-time adaptive signal control.",
        "Competitive Edge": "40% faster incident detection than TomTom HD Traffic, dramatically reducing emergency response times.",
        "Technical Differentiator": "Mesh networking for signal coordination enables seamless communication across diverse systems.",
        "Case Study": "12% faster emergency response times in Amsterdam (2023) via integrated traffic signal management.",
        "Application Sectors": ["Public Infrastructure", "Smart Cities", "Transport Planning"],
        "Performance Metrics": ["17% congestion drop", "94% detection rate", "112ms latency"]
    },
    "Crowdsourced Road Condition Awareness": {
        "Description": "12M+ IoT devices provide updates every 1.8s (industry avg: 15s). Achieving 94% pothole detection accuracy (Euro NCAP validated) through millimeter-wave analysis, now extended to include real-time weather and surface condition analytics.",
        "Competitive Edge": "Automated municipal reporting with 23% faster resolution times, driving proactive maintenance.",
        "Technical Differentiator": "Edge consensus protocol for data validation ensures high accuracy and reliability.",
        "Case Study": "€1.2M annual road maintenance savings for Milan by leveraging predictive road condition analytics.",
        "Application Sectors": ["Micromobility", "Public Infrastructure", "Fleet Operations"],
        "Performance Metrics": ["1.8s update speed", "94% accuracy", "12M devices"]
    },
    "Pedestrian Flow Optimization": {
        "Description": "38% crowd density reduction in hotspots using HeatFlow™ movement prediction. Integrated with 58 smart venues including Heathrow T5 and Camp Nou stadium, with enhanced real-time behavioral analytics to optimize pedestrian safety and flow.",
        "Competitive Edge": "0.1m movement tracking precision (patent pending) for ultra-fine crowd management.",
        "Technical Differentiator": "Multi-spectral sensor fusion technology combined with AI-driven analytics for dynamic flow control.",
        "Case Study": "19% foot traffic increase at London Westfield through optimized path routing and venue layout adjustments.",
        "Application Sectors": ["Urban Mobility", "Smart Cities", "Retail & Commercial Planning"],
        "Performance Metrics": ["38% density reduction", "0.1m precision", "58 venues"]
    },

    # Commercial & Industrial Applications
    "Connected Warehouses": {
        "Description": "53% faster inventory retrieval through centimeter-accurate RTLS. Processes 750K+ pallet movements/day with 99.98% accuracy using UWB positioning, now integrated with automated robotics for enhanced logistics operations.",
        "Competitive Edge": "Only system supporting Amazon Robotics' Kiva protocol, ensuring seamless automation.",
        "Technical Differentiator": "Sub-10cm 3D positioning system significantly reduces misplacement errors.",
        "Case Study": "31% efficiency gain at Zalando's Munich fulfillment center through precise asset tracking.",
        "Application Sectors": ["Logistics", "E-Commerce", "Supply Chain Management"],
        "Performance Metrics": ["53% speed gain", "99.98% accuracy", "750K/day capacity"]
    },
    "Supply Chain Monitoring": {
        "Description": "Predicts delays 26 minutes faster than SAP ECC. Reduced empty miles by 29% through adaptive load pooling. Blockchain audit trail cuts paperwork by 83% and now integrates real-time sensor data for holistic fleet management.",
        "Competitive Edge": "Smart contract-powered route compliance (Patent #11439876) ensuring end-to-end supply chain transparency.",
        "Technical Differentiator": "Multi-carrier optimization engine that dynamically adjusts routes based on live conditions.",
        "Case Study": "$4.7M annual fuel savings for Maersk land fleets via proactive route and load management.",
        "Application Sectors": ["Industrial Logistics", "Retail & E-Commerce", "Autonomous Drones"],
        "Performance Metrics": ["26min faster prediction", "29% mileage reduction", "83% less paperwork"]
    },

    # Developer Ecosystem & Business Performance
    "Developer Ecosystem": {
        "Description": "30% faster integration through 175 prebuilt API endpoints. 98.7% developer satisfaction (Gartner Peer Insights) with auto-GDPR compliance tools. Extensive SDK availability across C++, Flutter, iOS, Android, Javascript, and Qt empowers developers to rapidly integrate and innovate.",
        "Competitive Edge": "Free certification support for EU MaaS regulations, ensuring accelerated market adoption.",
        "Technical Differentiator": "Low-code workflow builder for fleet operations that simplifies complex integrations.",
        "Case Study": "47-day faster market entry for Bolt in the DACH region due to streamlined developer tools and cross-platform support.",
        "Application Sectors": ["Software Development", "Autonomous Vehicles", "Micromobility"],
        "Performance Metrics": ["175 APIs", "98.7% satisfaction", "30% speed gain"]
    },
    "Partnerships": {
        "Description": "49 certified partners including Siemens Mobility and Bosch Rexroth. 340% YoY growth in public sector contracts through the EU Innovation Fund, leveraging collaborative innovations and strategic data sharing.",
        "Competitive Edge": "Exclusive EUROCONTROL data sharing agreement that amplifies network effects.",
        "Technical Differentiator": "Interoperability framework for 15+ map formats ensures seamless partner integration.",
        "Case Study": "Madrid MaaS platform serving 2.8M monthly users, powered by strategic partner collaborations.",
        "Application Sectors": ["Urban Mobility", "Micromobility", "Fleet Partnerships"],
        "Performance Metrics": ["49 partners", "340% growth", "15+ formats"]
    },
    "Business Performance": {
        "Description": "173% YoY revenue growth (2023) with 94% client retention and a 28-minute SLA response time. Fleet deployments completed in <44 hours (industry avg: 336hrs), driven by a scalable cloud-native infrastructure that supports global operations.",
        "Competitive Edge": "Only profitable micromobility API provider since 2023 Q1, consistently delivering high-impact results.",
        "Technical Differentiator": "Auto-scaling Kubernetes infrastructure that dynamically adapts to operational demands.",
        "Case Study": "15K-unit Barcelona deployment for Bird in 264 hours, setting new industry benchmarks.",
        "Application Sectors": ["Autonomous Logistics", "Last-Mile Delivery", "Micromobility"],
        "Performance Metrics": ["173% growth", "94% retention", "44hr deployment"]
    },

    # Customer Base
    "Customers": {
        "Description": "92 enterprise clients including 7 Fortune 500 firms. 67% EU micromobility market share (IDC 2024) with a 4.8/5 NPS score and 100% contract renewal rate for municipal clients. Comprehensive client insights continuously drive product enhancements and market adaptation.",
        "Competitive Edge": "Preferred supplier for 9/10 top scooter operators, reflecting strong industry trust and market leadership.",
        "Technical Differentiator": "Multi-tenant policy management system that accommodates diverse client requirements.",
        "Case Study": "Voi's 12-city EU expansion in 163 days was achieved through dedicated client support and tailored solutions.",
        "Application Sectors": ["Micromobility", "Smart Cities", "E-Mobility Providers"],
        "Performance Metrics": ["92 clients", "67% share", "4.8 NPS"]
    }
}


# Data loading and preprocessing
def load_company_data():
    """Load data with session state updates for Magic Lane"""
    company_data = deepcopy(COMPANY_DATA)  # Create safe copy of original data
    
    # Check for user updates in session state
    if 'magic_lane_data' in st.session_state:
        ml_data = st.session_state.magic_lane_data
        for key in company_data:
            if key != 'Company' and key in ml_data:
                company_data[key][0] = ml_data[key]  # Update Magic Lane's scores
    
    df = pd.DataFrame(company_data)
    features = [col for col in df.columns if col not in ['Company', 'Total Score']]
    return df, features

def run_advanced_analysis(df, features):
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    # PCA Analysis
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]
    
    # Clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)  # Set n_init explicitly
    df['Cluster'] = kmeans.fit_predict(scaled_data)
    
    # Add cluster naming
    cluster_names = {
        0: "Niche Mapping Leaders",
        1: "Emerging Core Mappers",
        2: "Respected Players", 
        3: "Flex Innovators"
    }
    df['Cluster Name'] = df['Cluster'].map(cluster_names)
    
    # Calculate Z-scores for each feature
    for feature in features:
        df[f'{feature}_z'] = zscore(df[feature])
    
    return df
def create_market_position_plot(df):
    fig = px.scatter(df, 
                    x='PCA1', 
                    y='PCA2',
                    color='Cluster Name',  # Changed to show names
                    size='Total Score',
                    hover_data=['Company'],
                    title='Competitor Market Position Analysis',
                    template='plotly_white')
    
    # Add annotations for each cluster
    cluster_centers = df.groupby('Cluster Name')[['PCA1', 'PCA2']].mean().reset_index()
    for _, row in cluster_centers.iterrows():
        fig.add_annotation(
            x=row['PCA1'], 
            y=row['PCA2'], 
            text=row['Cluster Name'], 
            showarrow=True, 
            arrowhead=1
        )
    
    fig.update_layout(
        title_x=0.5,
        legend_title_text='Strategic Groups',
        height=600,
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True)
    )
    
    return fig

def create_radar_comparison(df, competitor, features):
    magic_lane_data = df[df['Company'] == 'Magic Lane'][features].values[0]
    competitor_data = df[df['Company'] == competitor][features].values[0]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=magic_lane_data,
        theta=features,
        fill='toself',
        name='Magic Lane'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=competitor_data,
        theta=features,
        fill='toself',
        name=competitor
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(magic_lane_data.max(), competitor_data.max())])),
        showlegend=True,
        title=f'Magic Lane vs {competitor} Comparison',
        height=600
    )
    return fig

def create_strength_weakness_analysis(df, competitor, features):
    ml_data = df[df['Company'] == 'Magic Lane'][features].iloc[0]
    comp_data = df[df['Company'] == competitor][features].iloc[0]
    
    # Calculate Z-score significance and raw difference
    sig_data = []
    for feature in features:
        ml_z = df[df['Company'] == 'Magic Lane'][f'{feature}_z'].values[0]
        comp_z = df[df['Company'] == competitor][f'{feature}_z'].values[0]
        sig_data.append({
            'Feature': feature,
            'Difference': ml_data[feature] - comp_data[feature],
            'ML_z': ml_z,
            'Comp_z': comp_z,
            'Sig_Strength': ml_z > 1.96,
            'Sig_Weakness': ml_z < -1.96
        })
    
    differences = pd.DataFrame(sig_data)
    
    strengths = differences[differences['Difference'] > 0].sort_values('Difference', ascending=False)
    weaknesses = differences[differences['Difference'] < 0].sort_values('Difference')
    
    return strengths, weaknesses

# -----------------------------
# New functions for dynamic tabs 4 & 5

# Add this new function to generate dynamic insights
def generate_strategic_insights(df, competitor, features, top_n=3):
    """Generate automated strategic insights based on Z-score differences and outliers"""
    insights = []
    z_diffs = []
    
    # Calculate Z-score differences
    for feature in features:
        ml_z = df[df['Company'] == 'Magic Lane'][f'{feature}_z'].values[0]
        comp_z = df[df['Company'] == competitor][f'{feature}_z'].values[0]
        z_diffs.append({
            'feature': feature,
            'z_diff': ml_z - comp_z,
            'ml_z': ml_z,  # Store individual Z-scores
            'comp_z': comp_z,  # Store competitor's Z-score
            'ml_score': df[df['Company'] == 'Magic Lane'][feature].values[0],
            'comp_score': df[df['Company'] == competitor][feature].values[0]
        })
    
    # Sort by absolute Z-score difference
    z_diffs_sorted = sorted(z_diffs, key=lambda x: abs(x['z_diff']), reverse=True)
    
    # Positive differentiators (Strengths)
    strengths = [item for item in z_diffs_sorted if item['z_diff'] > 0][:top_n]
    # Negative differentiators (Opportunities)
    weaknesses = [item for item in z_diffs_sorted if item['z_diff'] < 0][:top_n]
    
    # Generate strength insights
    if strengths:
        insights.append("### 🚀 Key Strategic Advantages")
        for item in strengths:
            strength_data = MAGIC_LANE_STRENGTHS.get(item['feature'], {})
            insights.append(f"""
**{item['feature']}** (Z-score +{abs(item['z_diff']):.1f}σ)
- *Competitive Edge*: {strength_data.get('Competitive Edge', 'Superior capabilities in this domain')}
- *Recommendation*: Leverage {item['feature']} leadership through {strength_data.get('Application Sectors', ['targeted marketing'])[0]} initiatives
- *Case Study*: {strength_data.get('Case Study', 'Proven track record of success')}
""")
    
    # Generate improvement insights
    if weaknesses:
        insights.append("### 🔍 Critical Improvement Opportunities")
        for item in weaknesses:
            strength_data = MAGIC_LANE_STRENGTHS.get(item['feature'], {})
            insights.append(f"""
**{item['feature']}** (Z-score gap: {item['z_diff']:.1f}σ)
- *Current Gap*: {item['comp_score'] - item['ml_score']:.1f} point deficit
- *Action Plan*: {strength_data.get('Technical Differentiator', 'Enhance capabilities through focused R&D')}
- *Target*: Aim for {item['ml_score'] + 5:.1f} points to reach significance threshold
""")
    
    # Identify outliers
    outliers = [item for item in z_diffs if abs(item['z_diff']) > 2.0]
    if outliers:
        insights.append("### 📌 Significant Outliers")
        insights.append("These features show statistically significant differences:")
        for item in outliers:
            insights.append(
                f"- **{item['feature']}**: Magic Lane Z-score {item['ml_z']:.1f}σ vs "  # Use stored ml_z
                f"{competitor} {item['comp_z']:.1f}σ"  # Use stored comp_z
            )
        insights.append("**Implications:** Consider strategic reallocation of resources based on these outlier metrics.")
    
    return "\n\n".join(insights)

# -----------------------------
def create_dynamic_score_drivers(df, competitor, features):
    """
    Tab 4: Display the raw difference (Magic Lane – Competitor) for each feature.
    """
    ml_row = df[df['Company'] == 'Magic Lane'].iloc[0]
    comp_row = df[df['Company'] == competitor].iloc[0]
    differences = {}
    for feature in features:
        differences[feature] = ml_row[feature] - comp_row[feature]
    differences_series = pd.Series(differences).sort_values(ascending=False)
    
    fig = px.bar(
        differences_series,
        x=differences_series.values,
        y=differences_series.index,
        orientation='h',
        title=f'Feature Score Drivers: Magic Lane vs {competitor}',
        labels={'x': 'Difference (Magic Lane - Competitor)', 'y': 'Feature'}
    )
    fig.update_layout(height=600)
    return fig

def create_dynamic_zscore_importance_plot(df, competitor, features):
    """
    Tab 5: Show the differential in Z-scores (Magic Lane – Competitor) for each feature.
    This highlights which features are statistically most different.
    """
    z_diffs = {}
    for feature in features:
        ml_z = df[df['Company'] == 'Magic Lane'][f'{feature}_z'].values[0]
        comp_z = df[df['Company'] == competitor][f'{feature}_z'].values[0]
        z_diffs[feature] = ml_z - comp_z
    z_diffs_series = pd.Series(z_diffs).sort_values(ascending=False)
    
    fig = px.bar(
        z_diffs_series,
        x=z_diffs_series.values,
        y=z_diffs_series.index,
        orientation='h',
        title=f'Z-score Differential (Magic Lane - {competitor})',
        labels={'x': 'Z-score Difference', 'y': 'Feature'}
    )
    fig.update_layout(height=600)
    return fig

# (The original functions for correlation heatmap and overall feature importance are preserved below for backward compatibility)
def create_correlation_heatmap(df):
    correlations = df[features].corrwith(df['Total Score']).sort_values(ascending=False)
    fig = px.bar(
        correlations,
        x=correlations.values,
        y=correlations.index,
        title='Feature Impact on Total Score',
        labels={'x': 'Pearson Correlation', 'y': 'Feature'}
    )
    fig.update_layout(height=600)
    return fig

def calculate_feature_importance(df, features):
    X = df[features]
    y = df['Total Score']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return importance

def create_feature_importance_plot(importance_df):
    fig = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        title='Strategic Feature Importance',
        labels={'x': 'Relative Impact', 'y': ''}
    )
    fig.update_layout(height=600)
    return fig

# -----------------------------
# Main App
# -----------------------------
def main():
    st.set_page_config(layout="wide", page_title="Magic Lane Competitive Analysis")
    
    
    # Initialize Magic Lane data in session state FIRST
    if 'magic_lane_data' not in st.session_state:
        default_ml = {k: v[0] for k, v in COMPANY_DATA.items() if k != 'Company'}
        st.session_state.magic_lane_data = default_ml

    # Load and process data
    df, features = load_company_data()
    df = run_advanced_analysis(df, features)
    
        # ====== NEW SIDEBAR CONTROLS ======
    # Magic Lane Data Editor 
    with st.sidebar.expander("⚙️ Magic Lane Configuration", expanded=False):
        st.image("https://www.magiclane.com/web/_next/static/media/symbol-purple.f0e48a33.svg", width=50)
        
        with st.form(key='ml_config'):
            st.header("Edit Magic Lane Scores")
            updated_data = {}
            
            # Create input fields for all scores
            for key in st.session_state.magic_lane_data:
                updated_data[key] = st.number_input(
                    f"{key} Score",
                    min_value=0.0,
                    max_value=1000.0,
                    value=float(st.session_state.magic_lane_data[key]),
                    key=f"ml_{key}"
                )
            
            # Form buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("💾 Save Changes"):
                    st.session_state.magic_lane_data = updated_data
                    st.success("Data updated!")
            with col2:
                if st.form_submit_button("🔄 Reset Defaults"):
                    default_ml = {k: v[0] for k, v in COMPANY_DATA.items() if k != 'Company'}
                    st.session_state.magic_lane_data = default_ml
                    st.success("Default values restored!")
    
    # Main dashboard header
    st.title("BATTLE READY Magic Lane Competitive Intelligence Dashboard against 28 competitors")
    st.image("https://www.counterpointresearch.com/_File/wp-content/uploads/2024/01/Counterpoint-Core-Scorecard-Location-Platform-Effectiveness-Index-2023-1.png", use_container_width =True)
    
    
    # Key Metrics (dynamic relative values)
    
    # Inject custom CSS to style the selectbox label
    st.markdown(
        """
        <style>
        label[for="select-competitor"] {
            font-size: 40px;
            font-weight: bold;
            color: #763;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    competitor = st.selectbox("Select Competitor", 
                                [c for c in COMPANY_DATA['Company'] if c != 'Magic Lane'],
                                key="select-competitor"
                            )
                            
    score_diff = (
        df[df['Company'] == 'Magic Lane']['Total Score'].iloc[0] -
        df[df['Company'] == competitor]['Total Score'].iloc[0]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Score Difference", f"{int(score_diff)} points")
    with col2:
        cluster_num = df[df['Company'] == competitor]['Cluster'].iloc[0]
        cluster_name = df[df['Company'] == competitor]['Cluster Name'].iloc[0]
        st.metric("Competitor Clusters Segment", 
                  f"Cluster {cluster_num}: {cluster_name}")
    st.markdown("""
**Cluster Definitions:**
- **Respected Players**: Established players with huge customer base & Map base across geographies
- **Emergent Core Mappers**: Mapping data specialists across nice use cases and sectors
- **Niche Mapping Leaders**: Players with a focus to disrupt a niche segment and grow  
- **Flex Innovators**: Specific focussed USP on selected parts of the tech
""")
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Market & Clusters Positions", "Competitive Comparison", "Generated Battle Cards agaist Competitor",
        "Score Drivers", "Strategic Priorities"
    ])
    
    with tab1:
        st.plotly_chart(create_market_position_plot(df), use_container_width=True)
    
    with tab2:
        st.plotly_chart(create_radar_comparison(df, competitor, features), use_container_width=True)
    
    with tab3:
        strengths, weaknesses = create_strength_weakness_analysis(df, competitor, features)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Validated Strengths")
            for _, row in strengths.iterrows():
                sig_icon = "🔥" if row['Sig_Strength'] else ""
                st.markdown(f"""
                    <div style='background-color:#e6f3ff;padding:15px;border-radius:5px;margin-bottom:10px'>
                        <h4>{row['Feature']} {sig_icon}</h4>
                        <p>+{row['Difference']:.1f} pts (Z-score +{row['ML_z']:.1f}σ)</p>
                        <div style='font-size:0.9em;color:#666'>
                            {MAGIC_LANE_STRENGTHS[row['Feature']]['Description']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Strategic Opportunities")
            for _, row in weaknesses.iterrows():
                imp_icon = "💡" if abs(row['Comp_z']) > 1.96 else ""
                st.markdown(f"""
                    <div style='background-color:#fff3e6;padding:15px;border-radius:5px;margin-bottom:10px'>
                        <h4>{row['Feature']} {imp_icon}</h4>
                        <p>{row['Difference']:.1f} pts (Z-score {row['ML_z']:.1f}σ)</p>
                        <div style='font-size:0.9em;color:#666'>
                            {MAGIC_LANE_STRENGTHS[row['Feature']]['Technical Differentiator']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    
    # Tab 4: Dynamic Score Drivers (relative raw differences)
    with tab4:
        st.plotly_chart(create_dynamic_score_drivers(df, competitor, features), use_container_width=True)
        st.markdown("""
            **Interpretation Guide:**
            - Bars indicate the point difference between Magic Lane and the selected competitor.
            - A positive value shows Magic Lane outperforms the competitor in that feature.
            - Use these insights to pinpoint key score drivers.
        """)
    
    # Tab 5: Dynamic Strategic Priorities (relative Z-score differences)
    # Modified Tab5 section in main()
    with tab5:
        st.plotly_chart(create_dynamic_zscore_importance_plot(df, competitor, features), use_container_width=True)
    
        # Dynamic Strategic Insights
        st.subheader("Automated Strategic Insights")
        insights = generate_strategic_insights(df, competitor, features)
    
        if insights:
            with st.expander("📈 Competitive Positioning Analysis", expanded=True):
                st.markdown(insights)
        else:
            st.warning("No significant insights found - consider deeper analysis")
        
        # Backward compatible section (preserves existing structure)
        st.markdown("""
        **Strategic Guidance:**
        - Z-score differences highlight statistically significant deviations
        - Larger differentials suggest areas of competitive strength/opportunity
        - Prioritize resource allocation based on strategic objectives
        """)
    
if __name__ == "__main__":
    main()
