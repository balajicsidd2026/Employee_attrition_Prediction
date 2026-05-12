import streamlit as st
import joblib
import pandas as pd
from catboost import Pool

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load(
"catboost_model.pkl"
)

# -----------------------------
# App Title
# -----------------------------
st.title(
"Employee Attrition Prediction"
)

# -----------------------------
# Categorical Inputs
# -----------------------------
role = st.selectbox(
"Role",
[
"Quality Inspector",
"Logistics Coordinator",
"Forklift Operator",
"Load Master",
"Dispatch Planner",
"Ground Handler",
"Warehouse Operator",
"Cargo Agent"
]
)

department = st.selectbox(
"Department",
[
"Warehouse",
"Ground Handling",
"Logistics Planning",
"Cargo Operations",
"Fleet Management",
"Quality & Compliance"
]
)

shift = st.selectbox(
"Shift",
[
"Morning",
"Afternoon",
"Night"
]
)

age_group = st.selectbox(
"Age Group",
[
"18-25",
"26-35",
"36-45",
"46-55",
"56+"
]
)

# -----------------------------
# Numeric Inputs
# -----------------------------
satisfaction = st.slider(
"Satisfaction Survey Score",
1,5,3
)

performance = st.slider(
"Performance Score",
1,5,3
)

tenure_months = st.number_input(
"Tenure Months",
min_value=0,
value=36
)

attendance = st.number_input(
"Attendance Rate Last 90 Days",
min_value=0,
max_value=100,
value=95
)

late_arrivals = st.number_input(
"Late Arrivals Last 90 Days",
min_value=0,
value=2
)

overtime = st.number_input(
"Overtime Hours Last Month",
min_value=0,
value=10
)

salary = st.number_input(
"Salary Percentile in Role",
min_value=0,
max_value=100,
value=50
)

promotions_received = st.number_input(
"Promotions Received",
min_value=0,
value=0
)

lateral_moves = st.number_input(
"Lateral Moves",
min_value=0,
value=1
)

promotion_days = st.number_input(
"Days Since Last Promotion",
min_value=0,
value=100
)

manager_tenure = st.number_input(
"Manager Tenure Months",
min_value=0,
value=24
)

training_hours = st.number_input(
"Training Hours Last Year",
min_value=0,
value=20
)


# -----------------------------
# Predict
# -----------------------------
if st.button(
"Predict Attrition"
):

    # Raw input data
    input_data = pd.DataFrame({

    'satisfaction_survey_score':[satisfaction],
    'performance_score':[performance],
    'salary_percentile_in_role':[salary],
    'overtime_hours_last_month':[overtime],
    'days_since_last_promotion':[promotion_days],
    'attendance_rate_last_90days':[attendance],
    'late_arrivals_last_90days':[late_arrivals],
    'manager_tenure_months':[manager_tenure],
    'training_hours_last_year':[training_hours],
    'tenure_months':[tenure_months],
    'lateral_moves':[lateral_moves],
    'promotions_received':[promotions_received],

    'role':[role],
    'department':[department],
    'shift':[shift],
    'age_group':[age_group]

    })


    # Force categorical as strings
    cat_cols = [
    'department',
    'role',
    'shift',
    'age_group'
    ]

    for col in cat_cols:
        input_data[col] = input_data[col].astype(str)


    # Create CatBoost Pool
    pred_pool = Pool(
    data=input_data,
    cat_features=cat_cols
    )


    # Predict probability
    prob_leave = model.predict_proba(
    pred_pool
    )[:,1][0]

    prob_stay = 1 - prob_leave


    # Best threshold from tuning
    threshold = 0.40


    # Decision
    if prob_leave >= threshold:

        st.error(
        "Employee is likely to LEAVE"
        )

    else:

        st.success(
        "Employee is likely to STAY"
        )


    # Show probabilities
    st.write(
    f"Probability of Leaving: {prob_leave:.2%}"
    )

    st.write(
    f"Probability of Staying: {prob_stay:.2%}"
    )