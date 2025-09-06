import pandas as pd
import re

ds_subject = pd.read_csv("dataset/listsubject2columns.csv")
ds_job_posting = pd.read_csv("dataset/select_LinkedInJob-2023.csv")

# Get the column names from ds
column_names = ds_subject.columns.tolist() + ds_job_posting.columns.tolist()

# Column selection
picking_columns_job_posting = [
    "Job_Title",
    "Job_Type",
    "Skill",
]

ds_subject = ds_subject["subject_name_en"]
ds_Job_Title = ds_job_posting["Job_Title"]
ds_Job_Type = ds_job_posting["Job_Type"]
ds_Job_Skill = ds_job_posting["Skill"]

# Clean Skill ds
ds_Job_Skill = ds_Job_Skill.str.lower()
skills = ds_Job_Skill.apply(lambda x: re.sub(r"[^a-z ,]", "", str(x)))
skills = skills.str.split(",").explode()
skills = skills.str.strip()
skills = skills[skills != ""].dropna().drop_duplicates()
ds_Job_Skill = skills.reset_index(drop=True)

# Clean Subject ds
subjects = ds_subject.str.lower()
subjects = subjects.apply(lambda x: re.sub(r"[^a-z ]", " ", str(x)))
subjects = subjects.apply(lambda x: re.sub(r"\s+", " ", str(x)).strip())
subjects = subjects.dropna().drop_duplicates()
ds_Subject = subjects.reset_index(drop=True)

# Clean Job Title ds
job_titles = ds_Job_Title.str.lower()
job_titles = job_titles.apply(lambda x: re.sub(r"[^a-z ]", " ", str(x)))
job_titles = job_titles.apply(lambda x: re.sub(r"\s+", " ", str(x)).strip())
job_titles = job_titles.dropna().drop_duplicates()
ds_Job_Title = job_titles.reset_index(drop=True)

# CLean Job Type ds
job_types = ds_Job_Type.str.lower()
job_types = job_types.apply(lambda x: re.sub(r"[^a-z ]", " ", str(x)))
job_types = job_types.apply(lambda x: re.sub(r"\s+", " ", str(x)).strip())
job_types = job_types.dropna().drop_duplicates()
ds_Job_Type = job_types.reset_index(drop=True)

ds_Subject.to_csv("dataset/cleaned_Subject.csv", index=False)
ds_Job_Title.to_csv("dataset/cleaned_Job_Title.csv", index=False)
ds_Job_Type.to_csv("dataset/cleaned_Job_Type.csv", index=False)
ds_Job_Skill.to_csv("dataset/cleaned_Job_Skill.csv", index=False)
print("Preprocessing completed and cleaned datasets saved.")
