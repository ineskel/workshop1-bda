import os
import time
import pandas as pd
import streamlit as st
import plotly.express as px
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# -----------------------------
# 1. Setup & DB connection
# -----------------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# -----------------------------
# 2. Queries definitions
# -----------------------------
QUERIES = {
    "1. Identification of traffic peaks and flow intensity": """
        WITH hourly_flow AS (
            SELECT 
                s.location_name,
                DATE_TRUNC('hour', r.record_time) AS hour_slot,
                COUNT(DISTINCT r.vehicle_id) AS vehicle_count
            FROM raw_traffic_readings r
            JOIN sensors s ON s.sensor_id = r.sensor_id
            GROUP BY s.location_name, hour_slot
        )
        SELECT *,
               RANK() OVER (PARTITION BY location_name ORDER BY vehicle_count DESC) AS rank_within_location
        FROM hourly_flow
        ORDER BY location_name, rank_within_location, hour_slot;
    """,
    "2. Movement efficiency assessment with slowdown detection": """
        WITH speed_stats AS (
            SELECT 
                s.sensor_id,
                s.location_name,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY r.speed) AS q1,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY r.speed) AS q3
            FROM raw_traffic_readings r
            JOIN sensors s ON s.sensor_id = r.sensor_id
            GROUP BY s.sensor_id, s.location_name
        )
        SELECT 
            r.sensor_id,
            s.location_name,
            r.record_time,
            r.speed,
            CASE 
                WHEN r.speed < q1 - 1.5*(q3 - q1) THEN 'Possible slowdown'
                WHEN r.speed > q3 + 1.5*(q3 - q1) THEN 'Abnormal high speed'
                ELSE 'Normal'
            END AS traffic_status
        FROM raw_traffic_readings r
        JOIN sensors s ON s.sensor_id = r.sensor_id
        JOIN speed_stats q ON q.sensor_id = s.sensor_id
        ORDER BY s.location_name, r.record_time;
    """,
    "3. Dynamic evaluation of traffic conditions": """
        SELECT
            s.location_name,
            r.record_time,
            AVG(r.speed) OVER (
                PARTITION BY s.sensor_id
                ORDER BY r.record_time
                ROWS BETWEEN 5 PRECEDING AND CURRENT ROW
            ) AS rolling_avg_speed
        FROM raw_traffic_readings r
        JOIN sensors s ON s.sensor_id = r.sensor_id
        ORDER BY s.location_name, r.record_time;
    """,
    "4. Density-speed correlation": """
            SELECT 
            s.location_name,
            COUNT(DISTINCT r.vehicle_id) AS vehicle_count,
            AVG(r.speed) AS avg_speed
        FROM raw_traffic_readings r
        JOIN sensors s ON s.sensor_id = r.sensor_id
        GROUP BY s.location_name
        ORDER BY vehicle_count DESC
    """,
    "5. Daily traffic trend": """
        WITH hourly AS (
            SELECT 
                EXTRACT(HOUR FROM record_time) AS hour_of_day,
                AVG(speed) AS avg_speed
            FROM raw_traffic_readings
            GROUP BY hour_of_day
        )
        SELECT 
            hour_of_day,
            avg_speed,
            LAG(avg_speed) OVER (ORDER BY hour_of_day) AS prev_hour_speed,
            ROUND(avg_speed - LAG(avg_speed) OVER (ORDER BY hour_of_day), 2) AS delta_speed
        FROM hourly
        ORDER BY hour_of_day;
    """,
    "6. Irregular patterns and possible incidents": """
        WITH stats AS (
            SELECT 
                s.sensor_id,
                AVG(r.speed) AS mean_speed,
                STDDEV(r.speed) AS std_speed
            FROM raw_traffic_readings r
            JOIN sensors s ON s.sensor_id = r.sensor_id
            GROUP BY s.sensor_id
        )
        SELECT 
            r.sensor_id,
            s.location_name,
            r.record_time,
            r.speed,
            ROUND((r.speed - st.mean_speed) / st.std_speed, 2) AS z_score,
            CASE 
                WHEN ABS((r.speed - st.mean_speed) / st.std_speed) > 2.0 THEN 'Irregular'
                ELSE 'Regular'
            END AS traffic_status
        FROM raw_traffic_readings r
        JOIN sensors s ON s.sensor_id = r.sensor_id
        JOIN stats st ON st.sensor_id = s.sensor_id
        ORDER BY ABS((r.speed - st.mean_speed) / st.std_speed) DESC;
    """,
    "7. Comparison of speed and time by road type": """
           SELECT 
        s.road_type,
        EXTRACT(HOUR FROM r.record_time)::int AS hour_of_day,
        AVG(r.speed) AS avg_speed
    FROM raw_traffic_readings r
    JOIN sensors s ON s.sensor_id = r.sensor_id
    GROUP BY s.road_type, hour_of_day
    ORDER BY s.road_type, hour_of_day;
    """
}

# -----------------------------
# 3. Streamlit UI
# -----------------------------
st.set_page_config(page_title="Traffic Analysis Dashboard", layout="wide")
st.markdown(
    """
    <style>
    /* Sidebar background */
    .css-1d391kg {  /* sidebar container class may vary by Streamlit version */
        background-color: #111111;  /* black */
        color: white;
    }
    /* Sidebar text and widgets */
    .css-1d391kg * {
        color: white !important;
    }

    /* Main content background */
    .css-18e3th9 {  /* main content container class may vary by Streamlit version */
        background-color: #f5f5f5;  /* light gray */
    }

    /* Streamlit headers/subheaders */
    h1, h2, h3, h4, h5, h6 {
        color: #222222;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Workshop 01 - Traffic Data Analytics Dashboard")

st.sidebar.title("Dashboard Navigation")
selected_query = st.sidebar.radio("Select desired analysis", list(QUERIES.keys()))
st.sidebar.markdown("---")
refresh_rate = 60
auto_refresh = st.sidebar.checkbox("Enable auto-refresh every 60 seconds", value=True)

# -----------------------------
# 4. Query execution and charts
# -----------------------------
def load_data(sql):
    return pd.read_sql_query(text(sql), engine)

while True:
    title = selected_query
    query = QUERIES[title]
    st.subheader(title)
    df = load_data(query)

    if df.empty:
        st.warning("No data returned.")
    else:
        if "flow" in title.lower():
            df["hour_slot"] = pd.to_datetime(df["hour_slot"])
            df = df.sort_values(by=["location_name", "hour_slot"])
            df["hour_of_day"] = df["hour_slot"].dt.hour
            all_locations = df["location_name"].unique()
            selected_locations = st.multiselect(
                "Select locations to display:",
                options=all_locations,
                default=all_locations,
                key=f"loc_select_{title}"
            )
            filtered_df = df[df["location_name"].isin(selected_locations)]
            fig = px.line(
                filtered_df,
                x="hour_slot",
                y="vehicle_count",
                color="location_name",
                title="Hourly Vehicle Flow per Location",
                markers=True
            )
            peaks = filtered_df[filtered_df["rank_within_location"] == 1]
            fig.add_scatter(
                x=peaks["hour_slot"],
                y=peaks["vehicle_count"],
                mode="markers+text",
                marker=dict(size=12, color="red", symbol="star"),
                text=peaks["hour_slot"].dt.strftime("%H:%M"),
                textposition="top center",
                name="Peak Hour"
            )

            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Vehicle Count",
                legend_title="Location",
                hovermode="x unified",
                legend=dict(
                    itemclick="toggle",
                    itemdoubleclick="toggleothers")
            )

            st.plotly_chart(fig, use_container_width=True)

        elif "slowdown" in title.lower():
            fig = px.scatter(df, x="record_time", y="speed", color="traffic_status", title="Speed Outlier Detection")
            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Speed",
                legend_title="Traffic status",
                hovermode="x unified",
                legend=dict(
                    itemclick="toggle",
                    itemdoubleclick="toggleothers")
            )
            st.plotly_chart(fig, use_container_width=True)
            summary = df.groupby("location_name")["traffic_status"].value_counts().unstack(fill_value=0)
            desired_order = ["Possible slowdown", "Abnormal high speed", "Normal"]
            summary = summary.reindex(columns=desired_order, fill_value=0) # ensure consistent order
            summary = summary.reset_index().rename(columns={"location_name": "Location Name"})
            st.subheader("Traffic Status Summary by Location")
            st.dataframe(summary)

        elif "dynamic" in title.lower():
            fig = px.line(df, x="record_time", y="rolling_avg_speed", color="location_name", title="Rolling Average Speed")
            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Average Speed",
                legend_title="Location name",
                hovermode="x unified",
                legend=dict(
                    itemclick="toggle",
                    itemdoubleclick="toggleothers")
            )
            st.plotly_chart(fig, use_container_width=True)

        elif "density" in title.lower():
            corr = df["vehicle_count"].corr(df["avg_speed"])
            st.metric(label="Correlation: Density vs Speed", value=round(corr, 2))
            display_df = df.rename(columns={
                "sensor_id": "Sensor ID",
                "vehicle_count": "Vehicle Count",
                "avg_speed": "Average Speed (km/h)"
            })
            
            st.dataframe(display_df.style.format({
                "Vehicle Count": "{:,.0f}",
                "Average Speed (km/h)": "{:.2f}"
            }))

        elif "trend" in title.lower():
            fig = px.line(df, x="hour_of_day", y="avg_speed", title="Average Speed Throughout the Day")
            st.plotly_chart(fig, use_container_width=True)

        elif "irregular" in title.lower():
            df["record_time"] = pd.to_datetime(df["record_time"])
            df["abs_z"] = df["z_score"].abs()
            fig = px.scatter(
                df,
                x="record_time",
                y="speed",
                color="abs_z",  # severity
                color_continuous_scale=["green", "yellow", "red"],  # low -> high
                facet_col="location_name",  # small multiples
                facet_col_wrap=1,           # wrap after 3 subplots
                title="Traffic Speed with Anomaly Severity by Location",
                hover_data=["z_score", "traffic_status"],
                height=1000
            )
            for loc in df["location_name"].unique():
                subset = df[df["location_name"] == loc]
                mean = subset["speed"].mean()
                std = subset["speed"].std()
                upper = mean + 1.7*std
                lower = mean - 1.7*std

            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Speed",
                coloraxis_colorbar=dict(title="Anomality severity"),
                showlegend=False,
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif "road type" in title.lower():
            df["hour_of_day"] = df["hour_of_day"].astype(int)
            fig = px.line(
                df,
                x="hour_of_day",
                y="avg_speed",
                color="road_type",
                title="Average Speed Across Road Types by Hour of Day",
                markers=True
            )
            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Average Speed (km/h)",
                legend_title="Road Type",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

    if not auto_refresh:
        break
    time.sleep(refresh_rate)
    st.rerun()
