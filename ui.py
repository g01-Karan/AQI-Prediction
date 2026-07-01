import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import customtkinter as ctk

# ================= MODEL (UNCHANGED LOGIC) =================
df = pd.read_csv('Cleaned_NSUT.csv')
X = df[['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Ozone']]
y = df['AQI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = r2_score(y_test, y_pred)
print("Model Accuracy (R² Score):", accuracy)
# =====================================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

ORANGE = "#ff7a30"
ORANGE_HOVER = "#e8641d"
SIDEBAR_BG = "#1c1c1e"
INPUT_BG = "#2a2a2d"


class AQIPredictorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AQI Predictor")
        self.geometry("1500x900")
        self.configure(fg_color="#111113")

        # Two-column layout: fixed sidebar (like the screenshot) + plot area
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.build_sidebar()
        self.build_plot_area()

    # ---------------- SIDEBAR ----------------
    def build_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=320, fg_color=SIDEBAR_BG, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)

        ctk.CTkLabel(
            sidebar, text="AQI\nPrediction",
            font=ctk.CTkFont(family="Sora", size=28, weight="bold"),
            text_color="white", justify="left"
        ).pack(anchor="w", padx=30, pady=(40, 5))

        ctk.CTkLabel(
            sidebar, text=f"Model R² Score: {accuracy:.4f}",
            font=ctk.CTkFont(family="Inter", size=12),
            text_color="#888888"
        ).pack(anchor="w", padx=30, pady=(0, 30))

        self.entries = {}
        fields = ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]

        for field in fields:
            ctk.CTkLabel(
                sidebar, text=field,
                font=ctk.CTkFont(family="Inter", size=14),
                text_color="white"
            ).pack(anchor="w", padx=30, pady=(10, 4))

            entry = ctk.CTkEntry(
                sidebar, placeholder_text="0.0", corner_radius=8,
                fg_color=INPUT_BG, border_color="#3a3a3d", border_width=1,
                height=38
            )
            entry.pack(padx=30, fill="x")
            self.entries[field] = entry

        self.result_label = ctk.CTkLabel(
            sidebar, text="",
            font=ctk.CTkFont(family="Sora", size=15, weight="bold"),
            text_color="#4ade80", justify="left"
        )
        self.result_label.pack(anchor="w", padx=30, pady=(30, 10))

        ctk.CTkButton(
            sidebar, text="Predict AQI", corner_radius=10, height=42,
            fg_color=ORANGE, hover_color=ORANGE_HOVER,
            font=ctk.CTkFont(family="Inter", size=14, weight="bold"),
            command=self.predict_aqi
        ).pack(padx=30, pady=(10, 20), fill="x")

    # ---------------- PLOT AREA ----------------
    def build_plot_area(self):
        plot_frame = ctk.CTkFrame(self, fg_color="white", corner_radius=0)
        plot_frame.grid(row=0, column=1, sticky="nsew")

        self.fig, self.ax = plt.subplots(figsize=(9, 7), dpi=100)
        self.fig.patch.set_facecolor("white")
        self.ax.set_facecolor("white")

        self.ax.scatter(y_test, y_pred, color="red", alpha=0.6, s=18, label="Actual vs Predicted")
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        self.ax.plot(lims, lims, color="#3b82f6", linewidth=2, label="Ideal Fit")

        self.ax.set_xlabel("Actual AQI")
        self.ax.set_ylabel("Predicted AQI")
        self.ax.set_title("AQI Prediction")
        self.ax.legend(loc="upper left")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)

        self.marker = None  # holds the highlighted prediction point

    # ---------------- LOGIC ----------------
    def predict_aqi(self):
        try:
            values = []
            for field in ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]:
                val = self.entries[field].get()
                if val.strip() == "":
                    raise ValueError(f"{field} is empty")
                values.append(float(val))

            input_df = pd.DataFrame([values], columns=["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"])
            prediction = model.predict(input_df)[0]

            self.result_label.configure(
                text=f"Predicted AQI\n{prediction:.2f}",
                text_color="#4ade80"
            )
            self.update_plot(prediction)

        except ValueError as e:
            self.result_label.configure(text=f"Error: {e}", text_color="#ff4d4d")

    def update_plot(self, prediction):
        if self.marker is not None:
            self.marker.remove()
        self.marker = self.ax.scatter(
            [prediction], [prediction], color="#3b82f6", s=140,
            edgecolors="white", linewidths=1.5, zorder=5, label="Your Prediction"
        )
        self.canvas.draw()


if __name__ == "__main__":
    app = AQIPredictorApp()
    app.mainloop()
