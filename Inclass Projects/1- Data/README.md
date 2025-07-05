# Project 1: Exploratory Data Analysis - Student Performance Dataset

## Overview
This project provides a hands-on experience with Exploratory Data Analysis (EDA) using a realistic student performance dataset. Students will learn to transform raw data into actionable insights through systematic analysis and visualization.

## Files
- `Project1_EDA_Student_Performance.ipynb` - Main Jupyter notebook with complete EDA workflow
- `Project1_EDA_Student_Performance_FILLIN.ipynb` - Fill-in version for classroom instruction
- `student_performance.csv` - Dataset with 50 student records
- `requirements.txt` - Python package dependencies
- `environment.yml` - Conda environment file
- `instructor_guide.md` - Teaching guide with solutions
- `README.md` - This file

## Dataset Description
The `student_performance.csv` contains the following features:

### Categorical Variables:
- **gender**: Male/Female
- **parental_education**: High School, Bachelor's, Master's
- **breakfast**: Yes/No (whether student eats breakfast)
- **lunch**: Yes/No (whether student eats lunch)

### Numerical Variables:
- **student_id**: Unique identifier
- **age**: Student age (17-19)
- **study_time**: Hours of study per day (1-4)
- **previous_score**: Previous academic score (60-98)
- **attendance_rate**: Class attendance rate (0.65-1.00)
- **final_grade**: Final grade (63-100)

## Setup Instructions

### Option 1: Using Conda (Recommended)
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate inclass-lab

# Launch Jupyter Notebook
jupyter notebook
```

### Option 2: Using pip
```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

### 3. Open the Project Notebook
- **For complete version**: Open `Project1_EDA_Student_Performance.ipynb`
- **For fill-in version**: Open `Project1_EDA_Student_Performance_FILLIN.ipynb`

## Learning Objectives

### Technical Skills:
- Data loading and inspection
- Missing value analysis
- Categorical vs numerical data handling
- Feature engineering
- Correlation analysis
- Data visualization with matplotlib and seaborn

### Analytical Skills:
- Systematic EDA approach
- Pattern recognition
- Insight generation
- Best practice identification

## Session Timeline (90 minutes)

### Kick-off (0-10 min)
- EDA best practices overview
- Project structure explanation

### Live Demo (10-55 min)
- Instructor walkthrough of notebook sections
- Real-time data exploration
- Visualization techniques

### Guided Exercise (55-80 min)
- Student practice with provided prompts
- Individual exploration time
- Instructor support

### Lightning Recap (80-90 min)
- Student presentations
- Best practices review
- Common pitfalls discussion

## Expected Outcomes
By the end of this session, students will be able to:
1. Perform systematic EDA on new datasets
2. Create meaningful visualizations
3. Identify key correlations and patterns
4. Generate actionable insights
5. Apply EDA best practices

## Tips for Success
- Follow the notebook structure step-by-step
- Experiment with different visualizations
- Document your insights clearly
- Ask questions during the guided exercise
- Be prepared to share your findings

## Next Steps
After completing this project, students will be ready for:
- More advanced statistical analysis
- Machine learning model development
- Feature engineering for predictive modeling
- Data preprocessing pipelines 