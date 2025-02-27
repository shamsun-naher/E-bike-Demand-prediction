# **Overall Project Flow**

---

1. **Data Engineer (DE)**:  
   * Prepares and enriches the raw scooter trip data.  
   * Focuses on combining trip data with external factors (weather, events, etc.).  
   * Leaves *spatial transformations* (like clustering or binning) to the ML Engineer.  
2. **Machine Learning Engineer (MLE)**:  
   * Uses the enriched data to build, test, and refine forecasting models.  
   * Performs spatial binning or clustering if needed (e.g., using k-means) to group locations.  
   * Simulates scenarios such as how a discount in a particular region might redistribute scooter trips.  
3. **Data Visualizer / Streamlit Developer (DV)**:  
   * Creates an **interactive dashboard** (e.g., in Streamlit) that shows the models’ forecasts and key insights.  
   * Adds simple user controls so teammates can see how demand changes if certain conditions (like a discount) are applied.

A **final 3-week hold-out** (unused for training) is reserved to verify how accurate our models really are in “future” situations.

---

---

# 

# **Making it Work Smoothly**

---

* **Start Simple, Evolve Gradually**:  
  * The Data Engineer produces daily demand data on Day 1, so the MLE can start baseline modeling immediately.  
  * Meanwhile, the Data Engineer moves on to gathering advanced features and the MLE begins exploring regional binning.  
* **Reserve the Final 3 Weeks**:  
  * The Data Engineer sets aside these days from the start.  
  * The MLE only tests on them at the very end to get a genuine “unseen future” performance check.  
* **Incremental Updates**:  
  * Each time the Data Engineer releases new features or data, the MLE can retrain/improve forecasts.  
  * The Data Visualizer updates the dashboard to reflect the latest predictions, ensuring everyone sees the newest results quickly.  
* **Team Communication**:  
  * Share updates daily or every other day so no one waits too long for new data, models, or visuals.

