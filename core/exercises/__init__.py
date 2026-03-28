from core.barbell import barbell_tracker
from core.exercises.bench_press import BenchPressChecker
from core.exercises.squat import SquatChecker
from core.exercises.deadlift import DeadliftChecker

EXERCISE_CHECKERS: dict = {
    "Bench Press": BenchPressChecker(barbell_tracker=barbell_tracker),
    "Back Squat":  SquatChecker(barbell_tracker=barbell_tracker),
    "Deadlift":    DeadliftChecker(barbell_tracker=barbell_tracker),
}
