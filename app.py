import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, session, jsonify
from werkzeug.utils import secure_filename
import io
import base64
from io import BytesIO
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK packages
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

app = Flask(__name__)
app.secret_key = 'eda_dashboard_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'txt'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# AI Chatbot functionality
class EdaChatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Define intent patterns and responses
        self.intents = {
            'greeting': {
                'patterns': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening'],
                'responses': [
                    "Hello! How can I help with your data analysis today?",
                    "Hi there! What would you like to explore in your dataset?",
                    "Hey! I'm here to assist with your data exploration."
                ]
            },
            'upload': {
                'patterns': ['upload', 'file', 'csv', 'data', 'dataset', 'import'],
                'responses': [
                    "You can upload your CSV file by clicking the 'Choose File' button on the home page.",
                    "To analyze your data, first upload a CSV file from the main page.",
                    "Upload your dataset using the file selector on the dashboard's main page."
                ]
            },
            'visualize': {
                'patterns': ['visualize', 'plot', 'graph', 'chart', 'histogram', 'bar', 'scatter', 'visualization'],
                'responses': [
                    "Go to the Visualizer tab to create plots like histograms, bar charts, scatter plots, and more.",
                    "Need visual insights? The Visualizer section lets you plot your data in various formats.",
                    "For data visualization, navigate to the Visualizer tab and select your preferred chart type."
                ]
            },
            'missing_values': {
                'patterns': ['missing', 'null', 'na', 'nan', 'empty', 'gaps'],
                'responses': [
                    "To handle missing values, go to the Missing Values tab where you can view and handle them.",
                    "You can detect and manage missing data in the Missing Values section.",
                    "The Missing Values tab allows you to fill or drop null values in your dataset."
                ]
            },
            'outliers': {
                'patterns': ['outlier', 'anomaly', 'extreme', 'unusual'],
                'responses': [
                    "To detect outliers, use the Outlier Detector tab which offers Z-score and IQR methods.",
                    "You can identify unusual data points using the outlier detection tools.",
                    "Check out the Outlier Detector for finding anomalies in your numerical data."
                ]
            },
            'filter': {
                'patterns': ['filter', 'search', 'query', 'find', 'condition'],
                'responses': [
                    "To filter your data, use the Filter & Search tab to apply various conditions.",
                    "Need to find specific records? Try the Filter & Search feature.",
                    "You can query your dataset using the Filter & Search functionality."
                ]
            },
            'export': {
                'patterns': ['export', 'save', 'download', 'extract'],
                'responses': [
                    "When you're done analyzing, visit the Export tab to download your data in various formats.",
                    "You can export your processed data as CSV, Excel, or JSON from the Export section.",
                    "To save your work, use the Export feature to download your analyzed dataset."
                ]
            },
            'stats': {
                'patterns': ['statistics', 'stats', 'summary', 'describe', 'mean', 'median', 'average'],
                'responses': [
                    "For statistical information, check the Statistical Info tab for detailed metrics.",
                    "You can view summary statistics in the Statistical Info section.",
                    "Need statistical measures? The Statistical Info tab provides comprehensive analysis."
                ]
            },
            'help': {
                'patterns': ['help', 'guide', 'how to', 'tutorial', 'instructions'],
                'responses': [
                    "I can help you navigate the dashboard. What specific feature do you need assistance with?",
                    "Need help? Ask me about uploading, visualizing, filtering, or any other feature.",
                    "I'm here to guide you through the EDA process. What would you like to learn about?"
                ]
            },
            'thanks': {
                'patterns': ['thanks', 'thank you', 'appreciate', 'helpful'],
                'responses': [
                    "You're welcome! Let me know if you need anything else.",
                    "Happy to help! Feel free to ask if you have more questions.",
                    "Anytime! I'm here to make your data analysis easier."
                ]
            },
            'features': {
                'patterns': ['features', 'capabilities', 'functions', 'what can you do', 'what can i do'],
                'responses': [
                    "This dashboard offers data upload, visualization, statistical analysis, missing value handling, outlier detection, filtering, and export capabilities.",
                    "You can upload data, create visualizations, analyze statistics, handle missing values, detect outliers, filter data, and export your results.",
                    "The EDA dashboard features include data summary, visualization tools, statistical analysis, data cleaning options, and export functionality."
                ]
            }
        }
        
        # Responses for data-specific questions
        self.data_analysis_responses = {
            'column_info': "To see information about column '{column}', check the Data Summary tab for basic stats or the Statistical Info tab for detailed metrics.",
            'missing_in_column': "You can view and handle missing values in '{column}' through the Missing Values tab.",
            'visualize_column': "To visualize '{column}', go to the Visualizer tab and select it as an X or Y variable depending on the plot type.",
            'outliers_in_column': "To detect outliers in '{column}', use the Outlier Detector tab and select this column.",
            'filter_column': "You can filter data based on '{column}' using the Filter & Search tab."
        }
        
        # Commands that trigger navigation suggestions
        self.navigation_commands = {
            'show_summary': {
                'patterns': ['show summary', 'data summary', 'overview', 'show overview'],
                'response': "To view the data summary, I recommend going to the Data Summary tab.",
                'url': '/data_summary'
            },
            'show_visualizer': {
                'patterns': ['create plot', 'make plot', 'create chart', 'show visualizer', 'make visualization'],
                'response': "To create visualizations, I recommend going to the Visualizer tab.",
                'url': '/visualizer'
            },
            'show_missing': {
                'patterns': ['handle missing', 'fix missing', 'missing values', 'show missing'],
                'response': "To handle missing values, I recommend going to the Missing Values tab.",
                'url': '/missing_values'
            },
            'show_outliers': {
                'patterns': ['find outliers', 'detect outliers', 'show outliers', 'outlier detection'],
                'response': "To detect outliers, I recommend going to the Outlier Detector tab.",
                'url': '/outlier_detector'
            },
            'show_filter': {
                'patterns': ['filter data', 'search data', 'query data', 'show filter'],
                'response': "To filter your data, I recommend going to the Filter & Search tab.",
                'url': '/filter_search'
            },
            'show_stats': {
                'patterns': ['show statistics', 'view stats', 'statistical information', 'show stats'],
                'response': "To view statistical information, I recommend going to the Statistical Info tab.",
                'url': '/statistical_info'
            },
            'show_export': {
                'patterns': ['export data', 'download data', 'save data', 'show export'],
                'response': "To export your data, I recommend going to the Export tab.",
                'url': '/export'
            }
        }
    
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Tokenize
        tokens = word_tokenize(text)
        # Remove punctuation and stop words, and lemmatize
        cleaned_tokens = []
        for token in tokens:
            if token.isalnum() and token not in self.stop_words:
                cleaned_tokens.append(self.lemmatizer.lemmatize(token))
        return cleaned_tokens
    
    def get_intent(self, user_input):
        # Preprocess user input
        processed_input = self.preprocess_text(user_input)
        
        # Check for navigation commands
        for command, info in self.navigation_commands.items():
            for pattern in info['patterns']:
                if pattern in user_input.lower():
                    return {
                        'type': 'navigation',
                        'command': command,
                        'response': info['response'],
                        'url': info['url']
                    }
        
        # Check for data-specific questions (if data is loaded)
        # This requires context from the session, so it's handled in the route
        
        # Match against intent patterns
        max_match = 0
        matched_intent = None
        
        for intent, data in self.intents.items():
            match_count = 0
            for pattern in data['patterns']:
                pattern_tokens = self.preprocess_text(pattern)
                for token in pattern_tokens:
                    if token in processed_input:
                        match_count += 1
            
            # Calculate match percentage based on user input length
            match_percentage = match_count / max(len(processed_input), 1) if processed_input else 0
            
            if match_percentage > 0.2 and match_percentage > max_match:  # Threshold for matching
                max_match = match_percentage
                matched_intent = intent
        
        if matched_intent:
            import random
            response = random.choice(self.intents[matched_intent]['responses'])
            return {
                'type': 'intent',
                'intent': matched_intent,
                'response': response
            }
        
        # Default response
        return {
            'type': 'unknown',
            'response': "I'm not sure how to help with that. Try asking about uploading data, creating visualizations, handling missing values, detecting outliers, or any other dashboard feature."
        }
    
    def check_data_specific_query(self, user_input, available_columns):
        """Check if the query is about specific columns in the data"""
        if not available_columns:
            return None
        
        # Check if any column name is mentioned
        for column in available_columns:
            if column.lower() in user_input.lower():
                # Check query type
                if any(word in user_input.lower() for word in ['missing', 'null', 'na', 'empty']):
                    return {
                        'type': 'data_specific',
                        'query_type': 'missing_in_column',
                        'column': column,
                        'response': self.data_analysis_responses['missing_in_column'].format(column=column)
                    }
                elif any(word in user_input.lower() for word in ['visualize', 'plot', 'graph', 'chart']):
                    return {
                        'type': 'data_specific',
                        'query_type': 'visualize_column',
                        'column': column,
                        'response': self.data_analysis_responses['visualize_column'].format(column=column)
                    }
                elif any(word in user_input.lower() for word in ['outlier', 'anomaly', 'extreme']):
                    return {
                        'type': 'data_specific',
                        'query_type': 'outliers_in_column',
                        'column': column,
                        'response': self.data_analysis_responses['outliers_in_column'].format(column=column)
                    }
                elif any(word in user_input.lower() for word in ['filter', 'search', 'find', 'where']):
                    return {
                        'type': 'data_specific',
                        'query_type': 'filter_column',
                        'column': column,
                        'response': self.data_analysis_responses['filter_column'].format(column=column)
                    }
                else:
                    return {
                        'type': 'data_specific',
                        'query_type': 'column_info',
                        'column': column,
                        'response': self.data_analysis_responses['column_info'].format(column=column)
                    }
        
        return None

# Initialize chatbot
chatbot = EdaChatbot()

@app.route('/chatbot_message', methods=['POST'])
def chatbot_message():
    if request.method == 'POST':
        data = request.json
        user_message = data.get('message', '')
        
        # Check if data is loaded and get columns
        available_columns = session.get('columns', [])
        
        # Check if message is about specific data columns
        data_specific = chatbot.check_data_specific_query(user_message, available_columns)
        
        if data_specific:
            response = data_specific
        else:
            # Get general intent response
            response = chatbot.get_intent(user_message)
        
        return jsonify(response)

# Include this in your templates
@app.route('/chatbot')
def chatbot_template():
    return render_template('chatbot.html')

# Add this to your layout template
# {% include 'chatbot.html' %}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read the CSV file
        try:
            df = pd.read_csv(filepath)
            # Store DataFrame info in session
            session['filename'] = filename
            session['filepath'] = filepath
            session['columns'] = df.columns.tolist()
            session['dtypes'] = df.dtypes.astype(str).to_dict()
            
            return redirect(url_for('data_summary'))
        except Exception as e:
            flash(f'Error reading CSV file: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('File type not allowed. Please upload a CSV file.')
        return redirect(url_for('index'))

@app.route('/data_summary')
def data_summary():
    if 'filepath' not in session:
        flash('Please upload a CSV file first')
        return redirect(url_for('index'))
    
    filepath = session['filepath']
    df = pd.read_csv(filepath)
    
    # Basic information
    rows, cols = df.shape
    columns = df.columns.tolist()
    dtypes = df.dtypes.astype(str).to_dict()
    
    # Missing values
    missing_values = df.isnull().sum().to_dict()
    missing_percentage = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
    
    # Preview data
    preview = df.head(10).to_html(classes='table table-striped table-hover', index=False)
    
    # Sample statistics
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    stats = None
    if numeric_columns:
        stats = df[numeric_columns].describe().to_html(classes='table table-striped table-hover')
    
    return render_template('data_summary.html', 
                          filename=session['filename'],
                          rows=rows,
                          cols=cols,
                          columns=columns,
                          dtypes=dtypes,
                          missing_values=missing_values,
                          missing_percentage=missing_percentage,
                          preview=preview,
                          stats=stats)

@app.route('/visualizer')
def visualizer():
    if 'filepath' not in session:
        flash('Please upload a CSV file first')
        return redirect(url_for('index'))
    
    filepath = session['filepath']
    df = pd.read_csv(filepath)
    
    # Get column names for the form
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    all_columns = df.columns.tolist()
    
    return render_template('visualizer.html',
                          numeric_columns=numeric_columns,
                          categorical_columns=categorical_columns,
                          all_columns=all_columns)

@app.route('/generate_plot', methods=['POST'])
def generate_plot():
    if 'filepath' not in session:
        flash('Please upload a CSV file first')
        return redirect(url_for('index'))
    
    filepath = session['filepath']
    df = pd.read_csv(filepath)
    
    plot_type = request.form.get('plot_type')
    x_column = request.form.get('x_column')
    y_column = request.form.get('y_column')
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    try:
        if plot_type == 'histogram':
            if x_column:
                plt.hist(df[x_column].dropna(), bins=30, alpha=0.7)
                plt.xlabel(x_column)
                plt.ylabel('Frequency')
                plt.title(f'Histogram of {x_column}')
        
        elif plot_type == 'bar':
            if x_column and y_column:
                counts = df.groupby(x_column)[y_column].mean().sort_values(ascending=False).head(15)
                counts.plot(kind='bar')
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.title(f'Bar Plot of {y_column} by {x_column}')
                plt.xticks(rotation=45)
        
        elif plot_type == 'scatter':
            if x_column and y_column:
                plt.scatter(df[x_column], df[y_column], alpha=0.5)
                plt.xlabel(x_column)
                plt.ylabel(y_column)
                plt.title(f'Scatter Plot of {x_column} vs {y_column}')
        
        elif plot_type == 'boxplot':
            if x_column:
                plt.boxplot(df[x_column].dropna())
                plt.xlabel(x_column)
                plt.title(f'Box Plot of {x_column}')
                plt.xticks([1], [x_column])
        
        elif plot_type == 'correlation':
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                correlation = numeric_df.corr()
                sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
                plt.title('Correlation Heatmap')
            else:
                plt.text(0.5, 0.5, "No numeric columns available for correlation", 
                         horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        
        # Save plot to a bytes buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Return the rendered template with the plot
        return render_template('plot_result.html', plot_data=plot_data)
    
    except Exception as e:
        plt.close()
        flash(f'Error generating plot: {str(e)}')
        return redirect(url_for('visualizer'))

@app.route('/filter_search')
def filter_search():
    if 'filepath' not in session:
        flash('Please upload a CSV file first')
        return redirect(url_for('index'))
    
    filepath = session['filepath']
    df = pd.read_csv(filepath)
    
    columns = df.columns.tolist()
    
    return render_template('filter_search.html', columns=columns)

@app.route('/apply_filter', methods=['POST'])
def apply_filter():
    if 'filepath' not in session:
        flash('Please upload a CSV file first')
        return redirect(url_for('index'))
    
    filepath = session['filepath']
    df = pd.read_csv(filepath)
    
    column = request.form.get('column')
    operation = request.form.get('operation')
    value = request.form.get('value')
    
    try:
        if operation == 'equals':
            filtered_df = df[df[column] == value]
        elif operation == 'contains':
            filtered_df = df[df[column].astype(str).str.contains(value, na=False)]
        elif operation == 'greater_than':
            filtered_df = df[df[column] > float(value)]
        elif operation == 'less_than':
            filtered_df = df[df[column] < float(value)]
        
        # Preview of filtered data
        filtered_preview = filtered_df.head(50).to_html(classes='table table-striped table-hover', index=False)
        filtered_count = len(filtered_df)
        
        return render_template('filtered_result.html', 
                              filtered_preview=filtered_preview,
                              filtered_count=filtered_count,
                              total_count=len(df))
    
    except Exception as e:
        flash(f'Error applying filter: {str(e)}')
        return redirect(url_for('filter_search'))

@app.route('/missing_values')
def missing_values():
    if 'filepath' not in session:
        flash('Please upload a CSV file first')
        return redirect(url_for('index'))
    
    filepath = session['filepath']
    df = pd.read_csv(filepath)
    
    # Missing values information
    missing_counts = df.isnull().sum().to_dict()
    missing_percentage = (df.isnull().sum() / len(df) * 100).round(2).to_dict()
    
    columns = df.columns.tolist()
    
    return render_template('missing_values.html', 
                          columns=columns,
                          missing_counts=missing_counts,
                          missing_percentage=missing_percentage)

@app.route('/handle_missing', methods=['POST'])
def handle_missing():
    if 'filepath' not in session:
        flash('Please upload a CSV file first')
        return redirect(url_for('index'))
    
    filepath = session['filepath']
    df = pd.read_csv(filepath)
    
    column = request.form.get('column')
    method = request.form.get('method')
    custom_value = request.form.get('custom_value')
    
    try:
        if method == 'drop':
            df = df.dropna(subset=[column])
            message = f"Dropped rows with missing values in column '{column}'"
        
        elif method == 'mean':
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna(df[column].mean())
                message = f"Filled missing values in '{column}' with mean: {df[column].mean()}"
            else:
                message = f"Cannot use mean for non-numeric column '{column}'"
        
        elif method == 'median':
            if pd.api.types.is_numeric_dtype(df[column]):
                df[column] = df[column].fillna(df[column].median())
                message = f"Filled missing values in '{column}' with median: {df[column].median()}"
            else:
                message = f"Cannot use median for non-numeric column '{column}'"
        
        elif method == 'mode':
            mode_value = df[column].mode()[0]
            df[column] = df[column].fillna(mode_value)
            message = f"Filled missing values in '{column}' with mode: {mode_value}"
        
        elif method == 'custom':
            df[column] = df[column].fillna(custom_value)
            message = f"Filled missing values in '{column}' with custom value: {custom_value}"
        
        # Save the modified DataFrame
        df.to_csv(filepath, index=False)
        flash(message)
        
        # Update preview
        preview = df.head(10).to_html(classes='table table-striped table-hover', index=False)
        missing_after = df[column].isnull().sum()
        
        return render_template('missing_result.html', 
                              preview=preview,
                              message=message,
                              missing_after=missing_after)
    
    except Exception as e:
        flash(f'Error handling missing values: {str(e)}')
        return redirect(url_for('missing_values'))

@app.route('/outlier_detector')
def outlier_detector():
    if 'filepath' not in session:
        flash('Please upload a CSV file first')
        return redirect(url_for('index'))
    
    filepath = session['filepath']
    df = pd.read_csv(filepath)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    return render_template('outlier_detector.html', columns=numeric_columns)

@app.route('/detect_outliers', methods=['POST'])
def detect_outliers():
    if 'filepath' not in session:
        flash('Please upload a CSV file first')
        return redirect(url_for('index'))
    
    filepath = session['filepath']
    df = pd.read_csv(filepath)
    
    column = request.form.get('column')
    method = request.form.get('method')
    
    try:
        if not pd.api.types.is_numeric_dtype(df[column]):
            flash(f"Column '{column}' is not numeric, cannot detect outliers")
            return redirect(url_for('outlier_detector'))
        
        if method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers = df[z_scores > 3]
            outlier_indices = z_scores > 3
            message = f"Detected {len(outliers)} outliers using Z-Score method (|z| > 3)"
        
        elif method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_indices = (df[column] < lower_bound) | (df[column] > upper_bound)
            outliers = df[outlier_indices]
            message = f"Detected {len(outliers)} outliers using IQR method"
        
        # Create box plot
        plt.figure(figsize=(10, 6))
        plt.boxplot(df[column].dropna())
        plt.title(f'Box Plot for {column} showing Outliers')
        plt.ylabel(column)
        plt.xticks([1], [column])
        
        # Save plot to a bytes buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Preview of outliers
        if len(outliers) > 0:
            outliers_preview = outliers.head(50).to_html(classes='table table-striped table-hover')
        else:
            outliers_preview = "No outliers detected"
        
        return render_template('outlier_result.html', 
                              plot_data=plot_data,
                              message=message,
                              outliers_preview=outliers_preview,
                              outlier_count=len(outliers),
                              total_count=len(df))
    
    except Exception as e:
        plt.close()
        flash(f'Error detecting outliers: {str(e)}')
        return redirect(url_for('outlier_detector'))

@app.route('/statistical_info')
def statistical_info():
    if 'filepath' not in session:
        flash('Please upload a CSV file first')
        return redirect(url_for('index'))
    
    filepath = session['filepath']
    df = pd.read_csv(filepath)
    
    # Basic statistics for numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        basic_stats = numeric_df.describe().to_html(classes='table table-striped table-hover')
    else:
        basic_stats = "No numeric columns available for statistics"
    
    # Additional statistics
    additional_stats = {}
    for column in numeric_df.columns:
        additional_stats[column] = {
            'skew': numeric_df[column].skew(),
            'kurtosis': numeric_df[column].kurtosis(),
            'variance': numeric_df[column].var(),
            'min': numeric_df[column].min(),
            'max': numeric_df[column].max(),
            'range': numeric_df[column].max() - numeric_df[column].min()
        }
    
    # Categorical columns
    categorical_stats = {}
    for column in df.select_dtypes(include=['object']).columns:
        value_counts = df[column].value_counts().head(10)
        categorical_stats[column] = {
            'unique_values': df[column].nunique(),
            'most_common': value_counts.index[0] if not value_counts.empty else 'N/A',
            'most_common_count': value_counts.iloc[0] if not value_counts.empty else 0,
            'value_counts': value_counts.to_dict()
        }
    
    return render_template('statistical_info.html',
                          basic_stats=basic_stats,
                          additional_stats=additional_stats,
                          categorical_stats=categorical_stats)

# Add these routes to your Flask app



@app.route('/export')
def export():
    if 'filepath' not in session:
        flash('Please upload a CSV file first')
        return redirect(url_for('index'))
    
    filepath = session['filepath']
    df = pd.read_csv(filepath)
    
    # Export options
    return render_template('export.html', filename=session['filename'])



@app.route('/download', methods=['POST'])
def download():
    if 'filepath' not in session:
        flash('Please upload a CSV file first')
        return redirect(url_for('index'))
    
    filepath = session['filepath']
    df = pd.read_csv(filepath)
    
    format_type = request.form.get('format_type')
    
    try:
        if format_type == 'csv':
            output = io.StringIO()
            df.to_csv(output, index=False)
            output.seek(0)
            
            return send_file(
                io.BytesIO(output.getvalue().encode()),
                mimetype='application/json',
                as_attachment=True,
                download_name=f"cleaned_{session['filename'].split('.')[0]}.json"
            )
    
    except Exception as e:
        flash(f'Error exporting data: {str(e)}')
        return redirect(url_for('export'))


if __name__ == '__main__':
    app.run(debug=True)