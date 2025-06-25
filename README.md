**📊 Advanced EDA Dashboard**

An intelligent and interactive **EDA Dashboard** that performs complete exploratory data analysis (EDA) from any CSV file. With the help of visualizations, statistical insights, outlier detection, and missing value handling, the dashboard also provides a **cleaned dataset for download** and a **smart chatbot** assistant to guide users.

**🚀 Live Demo**

🔗 eda-dashboard-1.onrender.com/

**🧠 Features**

✅ Upload any CSV file (small or large datasets)
✅ Get automatic summary statistics (mean, median, mode, count, etc.)
✅ Identify and visualize **missing values**
✅ Detect **outliers** using IQR or Z-score
✅ Generate:

Histograms

Correlation heatmaps

Box plots

Pair plots

Aggregated bar/pie charts
✅ **Fill missing values** (mean, median, mode options)
✅ **Drop unnecessary columns**
✅ Download cleaned CSV with one click
✅ Integrated **AI Chatbot** for:

Explaining features

Suggesting improvements

Guiding analysis
✅ Fast, scalable, responsive UI built using **Flask + Plotly / Seaborn**

**🛠 Tech Stack**

**🐍 Programming Languages**
Python 3.8+

SQL (for data queries)

**📦 Libraries & Frameworks**

**Data Analysis:** Pandas, NumPy

**Visualization:** Matplotlib, Seaborn, Plotly

**Statistical Analysis:** Scikit-learn, SciPy

**Outlier Detection:** IQR, Z-score, Isolation Forest

**📊 Dashboard & Web Frameworks**

Flask

**📁 File Handling & Export**

CSV upload/download (via pandas)

Cleaned file generation and export

**🔧 Tools & Utilities**

Git & GitHub

Jupyter / Google Colab (for prototyping)

Virtualenv / Pipenv (for env management)

**⚙️ Setup Instructions**

**✅ Prerequisites**

Python 3.8 or later

pip install virtualenv (optional but recommended)

**🔧 Local Installation**

# 1. Clone the repo
git clone github.com/Hemanthvedantam/EDA_Dashboard.git
cd eda-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Flask version)
python app.py

**🧪 Sample Usage**

Upload sample_data.csv

View:

Summary stats (mean, std, count)

Missing values heatmap

Correlation matrix

Handle missing values (auto fill or drop)

Detect and remove outliers

Download cleaned dataset

Ask chatbot: "What are the most correlated features?"

**🔒 Security & Limitations**

No data is stored – all analysis runs locally in memory

Handles datasets up to ~100MB smoothly

Chatbot is non-intrusive and runs in parallel

**📈 Future Enhancements**

Add automated report generation in PDF (using pandas-profiling or Sweetviz)

Multi-file upload support

Auto ML integration (recommend suitable models)

Theme toggle (light/dark mode)

Add ML model performance checker (post-cleaning)

**📬 Contact Me**

**📧 Email:** hemanthvedantam@gmail.com

**💼 LinkedIn:** linkedin.com/in/hemanthvedantam-813455280

**💻 GitHub:** github.com/Hemanthvedantam


