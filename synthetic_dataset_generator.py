"""Generate a linked synthetic dataset for the learning-practice pipeline.

Using a fixed random seed and the configured record counts, the generator
creates subject and student profiles, a hierarchy of topics and subtopics,
questions, exam results, practice sessions, and student-question interactions.
Synthetic learners have per-subject abilities and evolving per-subtopic
mastery; those values influence simulated response correctness and response
time. Session rows also include practice-mode and interleaving statistics.

Each dataset is written as a CSV in the current directory (``OUTPUT_DIR`` is
``"."``). The resulting files are intended as inputs for the project's model
training and recommendation workflows, not as real learner records. Running
``python synthetic_dataset_generator.py`` writes the datasets and reports the
number of generated records.
"""

import csv
import datetime
import json
import math
import random
import os

# ==========================================================
# CONFIGURATION (Scaled for Enhanced ML Training)
# ==========================================================

random.seed(42)
OUTPUT_DIR = "."

NUM_STUDENTS = 2000
SESSIONS_PER_STUDENT = 15
QUESTIONS_PER_SESSION = 10
QUESTIONS_PER_SUBTOPIC = 30

def save_csv(filename, data):
    """Write a list of row dictionaries to a CSV file.

    Empty datasets are reported and skipped. The CSV header is taken from the
    keys of the first row.
    """
    if not data:
        print(f"Skipping empty file: {filename}")
        return
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
    print(f"Generated: {filename} ({len(data)} records)")

def generate_enhanced_synthetic_data():
    """Generate and save the synthetic datasets used by the project.

    Student ability, topic mastery, practice sessions, and interaction
    outcomes are sampled using the configured dataset sizes and random seed.
    """
    print("\nStarting enhanced synthetic dataset generation...\n")
    current_time = datetime.datetime.now().replace(microsecond=0)

    # -----------------------------------------------------------------------
    # Curriculum Subjects and Student Profiles
    # -----------------------------------------------------------------------
    subjects = [
        {"subject_id": "SUBJ_MATH", "subject_name": "Mathematics", "educational_tier": "Both"},
        {"subject_id": "SUBJ_PHYS", "subject_name": "Physics", "educational_tier": "Both"},
        {"subject_id": "SUBJ_CHEM", "subject_name": "Chemistry", "educational_tier": "Both"}
    ]
    save_csv("subjects.csv", subjects)
    subject_ids = [s["subject_id"] for s in subjects]

    students = []
    for i in range(1, NUM_STUDENTS + 1):
        tier = random.choice(["K-12", "University"])
        grade = random.randint(1, 12) if tier == "K-12" else random.randint(1, 4)
        institution = random.choice(["K-12 Public", "K-12 Private"]) if tier == "K-12" else "Undergraduate"
        students.append({
            "student_id": f"STU_{i:04d}",
            "educational_tier": tier,
            "grade_level": grade,
            "institution_type": institution,
            "preferred_language": "English",
            "profile_creation_timestamp": (current_time - datetime.timedelta(days=365)).isoformat()
        })
    save_csv("students.csv", students)

    # -----------------------------------------------------------------------
    # Student Ability and Curriculum Structure
    # -----------------------------------------------------------------------
    student_base_abilities = {}
    for student in students:
        for subject in subjects:
            student_base_abilities[(student["student_id"], subject["subject_id"])] = random.gauss(0, 0.85)

    topics = []
    subtopic_ids = []
    subtopic_subject = {}
    subtopic_topic = {}

    for subject in subjects:
        for topic_index in range(1, 6):
            topic_id = f"{subject['subject_id']}_T{topic_index}"
            topic_name = f"{subject['subject_name']} Topic {topic_index}"
            for sub_index in range(1, 4):
                subtopic_id = f"{topic_id}_ST{sub_index}"
                subtopic_name = f"{topic_name} Subtopic {sub_index}"
                prerequisites = [f"{topic_id}_ST{sub_index - 1}"] if sub_index > 1 else []
                grade_level = random.randint(5, 12)
                topics.append({
                    "curriculum_id": f"CURR_{subject['subject_id']}",
                    "subject_id": subject["subject_id"],
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "subtopic_id": subtopic_id,
                    "subtopic_name": subtopic_name,
                    "grade_level": grade_level,
                    "prerequisite_topic_ids": json.dumps(prerequisites)
                })
                subtopic_ids.append(subtopic_id)
                subtopic_subject[subtopic_id] = subject["subject_id"]
                subtopic_topic[subtopic_id] = topic_id
    save_csv("topics.csv", topics)

    # -----------------------------------------------------------------------
    # Question Bank and Exam Results
    # -----------------------------------------------------------------------
    questions = []
    question_counter = 1
    for subtopic_id in subtopic_ids:
        subject_id = subtopic_subject[subtopic_id]
        for q_index in range(1, QUESTIONS_PER_SUBTOPIC + 1):
            difficulty = round(random.uniform(0.1, 0.9), 2)
            questions.append({
                "question_id": f"Q_{question_counter:06d}",
                "subtopic_id": subtopic_id,
                "question_text": f"Sample {subject_id} question for {subtopic_id}, variant {q_index}",
                "question_type": "Multiple Choice",
                "answer_options": json.dumps(["Option A", "Option B", "Option C", "Option D"]),
                "correct_answer": "Option A",
                "difficulty_level": difficulty,
                "concept_tag": f"TAG_{subtopic_id}",
                "estimated_solving_time_sec": int(30 + difficulty * 150),
                "marks_allocated": 1.0
            })
            question_counter += 1
    save_csv("questions.csv", questions)

    exam_results = []
    for student in students:
        for subject in subjects:
            ability = student_base_abilities[(student["student_id"], subject["subject_id"])]
            exam_percentage = max(10, min(100, 70 + (ability * 12) + random.gauss(0, 5)))
            exam_percentage = round(exam_percentage, 2)
            exam_results.append({
                "exam_id": f"EXAM_{student['student_id']}_{subject['subject_id']}",
                "student_id": student["student_id"],
                "subject_id": subject["subject_id"],
                "topic_id": f"{subject['subject_id']}_T1",
                "marks_obtained": exam_percentage,
                "total_marks": 100.0,
                "exam_date": (current_time - datetime.timedelta(days=90)).date().isoformat()
            })
    save_csv("student_exam_results.csv", exam_results)

    # -----------------------------------------------------------------------
    # Practice Sessions and Interaction Outcomes
    # -----------------------------------------------------------------------
    sessions = []
    interactions = []
    interaction_counter = 1
    
    questions_by_subtopic = {}
    for question in questions:
        st_id = question["subtopic_id"]
        questions_by_subtopic.setdefault(st_id, []).append(question)

    for student in students:
        student_id = student["student_id"]
        student_start = current_time - datetime.timedelta(days=180)
        
        # Track dynamic subtopic mastery per student across sessions
        subtopic_mastery = {st: 0.5 for st in subtopic_ids}
        subtopic_last_practiced = {st: student_start for st in subtopic_ids}
        performance_history = {st: [] for st in subtopic_ids}

        for session_index in range(1, SESSIONS_PER_STUDENT + 1):
            session_id = f"SESS_{student_id}_{session_index:03d}"
            session_start = student_start + datetime.timedelta(days=session_index * 5)
            subject = random.choice(subjects)
            subject_id = subject["subject_id"]
            
            eligible_subtopics = [st for st in subtopic_ids if subtopic_subject[st] == subject_id]
            selected_subtopics = random.sample(eligible_subtopics, min(5, len(eligible_subtopics)))
            
            session_questions = []
            for st_id in selected_subtopics:
                available = questions_by_subtopic[st_id]
                session_questions.extend(random.sample(available, min(2, len(available))))
            while len(session_questions) < QUESTIONS_PER_SESSION:
                st_id = random.choice(eligible_subtopics)
                session_questions.append(random.choice(questions_by_subtopic[st_id]))
            
            practice_mode = random.choice(["Interleaved", "Blocked"])
            if practice_mode == "Blocked":
                session_questions.sort(key=lambda q: q["subtopic_id"])
            else:
                random.shuffle(session_questions)

            question_ids = [q["question_id"] for q in session_questions]
            topic_sequence = [subtopic_topic[q["subtopic_id"]] for q in session_questions]
            topic_switches = sum(topic_sequence[i] != topic_sequence[i - 1] for i in range(1, len(topic_sequence)))
            
            entropy = (-sum((topic_sequence.count(t) / len(topic_sequence)) * math.log2(topic_sequence.count(t) / len(topic_sequence)) for t in set(topic_sequence)) if topic_sequence else 0)

            sessions.append({
                "session_id": session_id,
                "student_id": student_id,
                "session_practice_mode": practice_mode,
                "session_target_duration_min": 15,
                "session_total_questions": len(session_questions),
                "question_sequence": json.dumps(question_ids),
                "topic_switch_count": topic_switches,
                "interleaving_entropy": round(entropy, 3),
                "session_start_timestamp": session_start.isoformat()
            })

            for position, question in enumerate(session_questions, start=1):
                st_id = question["subtopic_id"]
                difficulty = float(question["difficulty_level"])
                base_ability = student_base_abilities[(student_id, subject_id)]
                
                history = performance_history[st_id]
                recent_acc = sum(history[-5:]) / len(history[-5:]) if history else 0.5
                current_mastery = subtopic_mastery[st_id]
                
                # Time since last practice in days
                delta_days = (session_start - subtopic_last_practiced[st_id]).total_seconds() / 86400.0
                forgetting_factor = math.exp(-0.1 * delta_days) # Ebbinghaus decay
                
                # Logit probability modeling
                logit = (
                    1.5 * base_ability +
                    2.0 * current_mastery +
                    1.0 * recent_acc -
                    2.2 * difficulty +
                    0.5 * forgetting_factor
                )
                probability = 1.0 / (1.0 + math.exp(-logit))
                is_correct = 1 if random.random() < probability else 0

                # Update mastery with learning rate
                learning_rate = 0.1
                subtopic_mastery[st_id] = max(0.0, min(1.0, current_mastery + learning_rate * (is_correct - difficulty)))
                subtopic_last_practiced[st_id] = session_start
                history.append(is_correct)

                # Response time modeling
                base_time = 30 + difficulty * 120
                response_time = int(random.lognormvariate(math.log(base_time) - 0.2 * base_ability, 0.25) * 1000)
                response_time = max(3000, min(response_time, 300000))
                
                interaction_time = session_start + datetime.timedelta(minutes=(position - 1) * 1.5)

                interactions.append({
                    "interaction_id": f"INT_{interaction_counter:08d}",
                    "session_id": session_id,
                    "student_id": student_id,
                    "question_id": question["question_id"],
                    "attempt_number": 1,
                    "student_answer": "Option A" if is_correct else random.choice(["Option B", "Option C", "Option D"]),
                    "is_correct": is_correct,
                    "marks_obtained": question["marks_allocated"] if is_correct else 0.0,
                    "response_time_ms": response_time,
                    "interaction_timestamp": interaction_time.isoformat()
                })
                interaction_counter += 1

    save_csv("practice_sessions.csv", sessions)
    save_csv("student_interactions.csv", interactions)
    print("\nEnhanced synthetic data generation completed successfully!")

if __name__ == "__main__":
    generate_enhanced_synthetic_data()
