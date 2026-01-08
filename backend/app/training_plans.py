"""
Training plan generator module.
Generates personalized workout plans based on age, objective, level, and frequency.
"""

from typing import List, Dict
from pydantic import BaseModel


class TrainingPlanRequest(BaseModel):
    age: int
    objective: str  # weight_loss, muscle_gain, fitness
    level: str  # beginner, intermediate, advanced
    frequency: int  # sessions per week


class Exercise(BaseModel):
    name: str
    sets: int
    reps: str
    rest: str = "60-90s"


class DayPlan(BaseModel):
    exercises: List[Exercise]


# Exercise database
EXERCISES = {
    "cardio": [
        {"name": "Course à pied", "sets": 1, "reps": "20-30 min", "rest": "N/A"},
        {"name": "Saut à la corde", "sets": 3, "reps": "2-3 min", "rest": "30s"},
        {"name": "Burpees", "sets": 3, "reps": "10-15", "rest": "60s"},
        {"name": "Mountain Climbers", "sets": 3, "reps": "30-45s", "rest": "30s"},
    ],
    "strength": [
        {"name": "Squats", "sets": 4, "reps": "8-12", "rest": "90s"},
        {"name": "Pompes", "sets": 4, "reps": "8-15", "rest": "60s"},
        {"name": "Planche", "sets": 3, "reps": "30-60s", "rest": "60s"},
        {"name": "Fentes", "sets": 3, "reps": "10-12 par jambe", "rest": "60s"},
        {"name": "Développé couché", "sets": 4, "reps": "8-10", "rest": "90s"},
        {"name": "Tractions", "sets": 3, "reps": "5-10", "rest": "90s"},
        {"name": "Soulevé de terre", "sets": 4, "reps": "6-8", "rest": "120s"},
    ],
    "core": [
        {"name": "Planche", "sets": 3, "reps": "30-60s", "rest": "60s"},
        {"name": "Crunch", "sets": 3, "reps": "15-20", "rest": "45s"},
        {"name": "Leg Raises", "sets": 3, "reps": "12-15", "rest": "60s"},
        {"name": "Russian Twists", "sets": 3, "reps": "20-30", "rest": "45s"},
    ],
    "flexibility": [
        {"name": "Étirements dynamiques", "sets": 1, "reps": "10-15 min", "rest": "N/A"},
        {"name": "Yoga flow", "sets": 1, "reps": "15-20 min", "rest": "N/A"},
    ]
}


def adjust_for_age(age: int, base_reps: str, base_sets: int) -> tuple:
    """Adjust reps and sets based on age."""
    if age < 18:
        # Adolescents: reduce intensity
        if isinstance(base_reps, str) and "min" in base_reps:
            return base_sets, base_reps
        reps_num = int(base_reps.split("-")[0]) if "-" in base_reps else int(base_reps)
        return base_sets, f"{max(5, int(reps_num * 0.8))}-{max(8, int(reps_num * 0.9))}"
    elif age > 50:
        # Seniors: reduce intensity, focus on form
        if isinstance(base_reps, str) and "min" in base_reps:
            return base_sets, base_reps
        reps_num = int(base_reps.split("-")[0]) if "-" in base_reps else int(base_reps)
        return max(2, base_sets - 1), f"{max(5, int(reps_num * 0.7))}-{max(10, int(reps_num * 0.85))}"
    return base_sets, base_reps


def adjust_for_level(level: str, exercises: List[Dict]) -> List[Dict]:
    """Adjust exercises based on fitness level."""
    adjusted = []
    for ex in exercises:
        ex_copy = ex.copy()
        if level == "beginner":
            if isinstance(ex["reps"], str) and "-" in ex["reps"]:
                parts = ex["reps"].split("-")
                ex_copy["reps"] = f"{parts[0]}-{int(parts[1]) * 0.7}"
            ex_copy["sets"] = max(2, ex["sets"] - 1)
            ex_copy["rest"] = "90-120s" if "s" in ex.get("rest", "") else ex.get("rest", "90s")
        elif level == "advanced":
            if isinstance(ex["reps"], str) and "-" in ex["reps"]:
                parts = ex["reps"].split("-")
                ex_copy["reps"] = f"{int(parts[0]) * 1.2}-{int(parts[1]) * 1.3}"
            ex_copy["sets"] = ex["sets"] + 1
            ex_copy["rest"] = "45-60s" if "s" in ex.get("rest", "") else ex.get("rest", "60s")
        adjusted.append(ex_copy)
    return adjusted


def generate_weight_loss_plan(age: int, level: str, frequency: int) -> List[DayPlan]:
    """Generate a weight loss focused plan."""
    plan = []
    
    # High cardio focus
    cardio_exercises = adjust_for_level(level, EXERCISES["cardio"])
    strength_exercises = adjust_for_level(level, EXERCISES["strength"][:3])  # Full body focus
    core_exercises = adjust_for_level(level, EXERCISES["core"][:2])
    
    for day in range(frequency):
        exercises = []
        
        # Day 1, 3, 5: Cardio + Full Body
        if day % 2 == 0:
            # Cardio warm-up
            cardio = cardio_exercises[day % len(cardio_exercises)]
            sets, reps = adjust_for_age(age, cardio["reps"], cardio["sets"])
            exercises.append(Exercise(name=cardio["name"], sets=sets, reps=reps, rest=cardio["rest"]))
            
            # Full body strength
            for ex in strength_exercises:
                sets, reps = adjust_for_age(age, ex["reps"], ex["sets"])
                exercises.append(Exercise(name=ex["name"], sets=sets, reps=reps, rest=ex["rest"]))
        
        # Day 2, 4, 6: HIIT + Core
        else:
            # HIIT circuit
            exercises.append(Exercise(name="Burpees", sets=4, reps="30s", rest="30s"))
            exercises.append(Exercise(name="Mountain Climbers", sets=4, reps="30s", rest="30s"))
            exercises.append(Exercise(name="Squats Sautés", sets=3, reps="15-20", rest="45s"))
            
            # Core
            for ex in core_exercises:
                sets, reps = adjust_for_age(age, ex["reps"], ex["sets"])
                exercises.append(Exercise(name=ex["name"], sets=sets, reps=reps, rest=ex["rest"]))
        
        plan.append(DayPlan(exercises=exercises))
    
    return plan


def generate_muscle_gain_plan(age: int, level: str, frequency: int) -> List[DayPlan]:
    """Generate a muscle gain focused plan."""
    plan = []
    
    # Split routine based on frequency
    if frequency <= 3:
        # Full body each session
        strength_exercises = adjust_for_level(level, EXERCISES["strength"])
        core_exercises = adjust_for_level(level, EXERCISES["core"][:2])
        
        for day in range(frequency):
            exercises = []
            # Select 5-6 compound movements
            selected = strength_exercises[:6]
            for ex in selected:
                sets, reps = adjust_for_age(age, ex["reps"], ex["sets"])
                exercises.append(Exercise(name=ex["name"], sets=sets, reps=reps, rest=ex["rest"]))
            
            # Add core at end
            for ex in core_exercises:
                sets, reps = adjust_for_age(age, ex["reps"], ex["sets"])
                exercises.append(Exercise(name=ex["name"], sets=sets, reps=reps, rest=ex["rest"]))
            
            plan.append(DayPlan(exercises=exercises))
    
    else:
        # Push/Pull/Legs split
        push_exercises = ["Pompes", "Développé couché", "Dips"]
        pull_exercises = ["Tractions", "Rowing", "Face Pull"]
        legs_exercises = ["Squats", "Soulevé de terre", "Fentes"]
        
        split_map = {
            0: push_exercises,
            1: pull_exercises,
            2: legs_exercises,
        }
        
        for day in range(frequency):
            exercises = []
            split_type = day % 3
            exercise_names = split_map[split_type]
            
            for name in exercise_names:
                # Find exercise in database
                ex = next((e for e in EXERCISES["strength"] if e["name"] == name), None)
                if ex:
                    sets, reps = adjust_for_age(age, ex["reps"], ex["sets"])
                    exercises.append(Exercise(name=ex["name"], sets=sets, reps=reps, rest=ex["rest"]))
            
            plan.append(DayPlan(exercises=exercises))
    
    return plan


def generate_fitness_plan(age: int, level: str, frequency: int) -> List[DayPlan]:
    """Generate a general fitness maintenance plan."""
    plan = []
    
    # Balanced mix of cardio, strength, and flexibility
    cardio_exercises = adjust_for_level(level, EXERCISES["cardio"][:2])
    strength_exercises = adjust_for_level(level, EXERCISES["strength"][:4])
    core_exercises = adjust_for_level(level, EXERCISES["core"])
    flexibility_exercises = EXERCISES["flexibility"]
    
    for day in range(frequency):
        exercises = []
        
        # Rotate focus
        if day % 3 == 0:
            # Cardio + Core
            cardio = cardio_exercises[0]
            sets, reps = adjust_for_age(age, cardio["reps"], cardio["sets"])
            exercises.append(Exercise(name=cardio["name"], sets=sets, reps=reps, rest=cardio["rest"]))
            
            for ex in core_exercises[:2]:
                sets, reps = adjust_for_age(age, ex["reps"], ex["sets"])
                exercises.append(Exercise(name=ex["name"], sets=sets, reps=reps, rest=ex["rest"]))
        
        elif day % 3 == 1:
            # Strength
            for ex in strength_exercises:
                sets, reps = adjust_for_age(age, ex["reps"], ex["sets"])
                exercises.append(Exercise(name=ex["name"], sets=sets, reps=reps, rest=ex["rest"]))
        
        else:
            # Flexibility + Light Strength
            flex = flexibility_exercises[0]
            exercises.append(Exercise(name=flex["name"], sets=flex["sets"], reps=flex["reps"], rest=flex["rest"]))
            
            for ex in strength_exercises[:2]:
                sets, reps = adjust_for_age(age, ex["reps"], ex["sets"])
                exercises.append(Exercise(name=ex["name"], sets=sets, reps=reps, rest=ex["rest"]))
        
        plan.append(DayPlan(exercises=exercises))
    
    return plan


def generate_training_plan(request: TrainingPlanRequest) -> Dict:
    """Main function to generate training plan based on request."""
    if request.objective == "weight_loss":
        plan = generate_weight_loss_plan(request.age, request.level, request.frequency)
    elif request.objective == "muscle_gain":
        plan = generate_muscle_gain_plan(request.age, request.level, request.frequency)
    elif request.objective == "fitness":
        plan = generate_fitness_plan(request.age, request.level, request.frequency)
    else:
        raise ValueError(f"Unknown objective: {request.objective}")
    
    return {
        "success": True,
        "plan": [day.dict() for day in plan],
        "summary": {
            "objective": request.objective,
            "level": request.level,
            "frequency": request.frequency,
            "total_days": len(plan)
        }
    }
