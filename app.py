from flask import Flask, render_template, request, jsonify, redirect, url_for
from vectorly import (IntegralSolver, FluxCalculator, FlowCalculator, 
                 CirculationCalculator, SolveWorkIntegral, 
                 ConservativeChecker, PotentialFunctionCalculator)
import json
import re
from difflib import SequenceMatcher
import logging
from math import isfinite

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced operation mapping with categories
OPERATION_MAP = {
    "Line Integral": {
        "key": "line_integral",
        "keywords": ["line integral", "∫", "integral", "curve"]
    },
    "Flux": {
        "key": "flux",
        "keywords": ["flux", "surface integral", "∬"]
    },
    "Flow": {
        "key": "flow",
        "keywords": ["flow", "fluid", "velocity field"]
    },
    "Circulation": {
        "key": "circulation",
        "keywords": ["circulation", "∮", "closed loop"]
    },
    "Work Integral": {
        "key": "work_integral",
        "keywords": ["work", "force", "displacement"]
    },
    "Conservativity Check": {
        "key": "conservativity",
        "keywords": ["conservative", "curl", "∇×"]
    },
    "Potential Function": {
        "key": "potential_function",
        "keywords": ["potential", "φ", "scalar field"]
    }
}

# Load all problem datasets with caching
PROBLEMS = {}

def load_problems():
    """Load all problem datasets with error handling and caching"""
    if PROBLEMS:  # Return cached data if already loaded
        return PROBLEMS

    datasets = {
        'line_integral': 'lineintegrals.json',
        'flux': 'flux.json',
        'flow': 'flow.json',
        'circulation': 'circulation.json',
        'work_integral': 'workintegrals.json',
        'conservativity': 'conservativity.json',
        'potential_function': 'potentialfunctions.json'
    }

    for operation, filename in datasets.items():
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                PROBLEMS[operation] = data['problems'] if 'problems' in data else data
                logger.info(f"Loaded {len(PROBLEMS[operation])} problems for {operation}")
        except FileNotFoundError:
            logger.warning(f"File not found: {filename}")
            PROBLEMS[operation] = []
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {filename}")
            PROBLEMS[operation] = []

    return PROBLEMS

# Initialize problem data
load_problems()

def similar(a, b):
    """Improved similarity measure with keyword boosting"""
    # Convert inputs to strings if they aren't already
    def to_string(obj):
        if isinstance(obj, dict):
            # For dictionary vector fields, convert to string format
            if 'i' in obj and 'j' in obj:
                i_comp = obj.get('i', '0')
                j_comp = obj.get('j', '0')
                k_comp = obj.get('k', '0')
                return f"{i_comp}i + {j_comp}j + {k_comp}k"
            else:
                return str(obj)
        elif isinstance(obj, (list, tuple)):
            return ' '.join(str(item) for item in obj)
        else:
            return str(obj)
    
    a_str = to_string(a)
    b_str = to_string(b)
    
    return SequenceMatcher(None, a_str.lower(), b_str.lower()).ratio()

def extract_key_components(text):
    """Extract mathematical components from problem text"""
    components = {
        'vector_field': None,
        'curve': None,
        'limits': None,
        'keywords': []
    }
    
    # Match vector fields (e.g., F = xi + yj)
    vector_field_match = re.search(r'([Ff]\s*=\s*[^.,;]+)', text)
    if vector_field_match:
        components['vector_field'] = vector_field_match.group(1)
    
    # Match curves/surfaces (e.g., "along y = x^2")
    curve_match = re.search(r'(along|curve|path|surface)\s*([^.,;]+)', text, re.IGNORECASE)
    if curve_match:
        components['curve'] = curve_match.group(2)
    
    # Match limits (e.g., "from 0 to 1")
    limits_match = re.search(r'(from|between)\s*([^\s]+)\s*(to|and)\s*([^\s.,;]+)', text, re.IGNORECASE)
    if limits_match:
        components['limits'] = (limits_match.group(2), limits_match.group(4))
    
    return components

def find_best_match(user_input, problems, operation_key):
    """Enhanced problem matching with mathematical component analysis"""
    if not problems:
        return None

    # Extract components from user input
    user_components = extract_key_components(user_input)
    best_match = None
    highest_score = 0
    
    # Get operation keywords for boosting
    op_keywords = [op['keywords'] for op in OPERATION_MAP.values() if op['key'] == operation_key][0]
    
    for problem in problems:
        score = 0
        
        # Create combined problem text from all available fields
        problem_text = ' '
        for field in ['problem_statement', 'question', 'vector_field', 'curve', 'path']:
            if field in problem:
                field_value = problem[field]
                if field == 'vector_field' and isinstance(field_value, dict):
                    # Convert dictionary vector field to string format
                    i_comp = field_value.get('i', '0')
                    j_comp = field_value.get('j', '0')
                    k_comp = field_value.get('k', '0')
                    field_value = f"{i_comp}i + {j_comp}j + {k_comp}k"
                problem_text += ' ' + str(field_value)
        
        # Base similarity score
        score += 0.6 * similar(user_input, problem_text)
        
        # Boost for matching vector fields
        if user_components['vector_field'] and 'vector_field' in problem:
            score += 0.3 * similar(user_components['vector_field'], problem['vector_field'])
        
        # Boost for matching curves/paths
        if user_components['curve']:
            curve_fields = ['curve', 'path', 'parametric_equations']
            for field in curve_fields:
                if field in problem:
                    score += 0.2 * similar(user_components['curve'], problem[field])
        
        # Boost for operation keywords
        for keyword in op_keywords:
            if keyword in user_input.lower():
                score += 0.1
        
        # Update best match
        if score > highest_score:
            highest_score = score
            best_match = problem
    
    logger.debug(f"Best match score: {highest_score:.2f}")
    return best_match if highest_score > 0.5 else None

def get_solver_class(operation_key):
    """Get the appropriate solver class with error handling"""
    solvers = {
        'line_integral': IntegralSolver,
        'flux': FluxCalculator,
        'flow': FlowCalculator,
        'circulation': CirculationCalculator,
        'work_integral': SolveWorkIntegral,
        'conservativity': ConservativeChecker,
        'potential_function': PotentialFunctionCalculator
    }
    return solvers.get(operation_key)

def format_result(result, operation_key):
    """Format the result based on operation type"""
    if result is None:
        return None
    
    # Handle numeric results
    if isinstance(result, (int, float)) and isfinite(result):
        return round(result, 6)
    
    # Handle symbolic results
    if hasattr(result, 'evalf'):
        try:
            float_val = float(result.evalf())
            return round(float_val, 6) if isfinite(float_val) else str(result)
        except:
            return str(result)
    
    # Handle conservativity results
    if operation_key == 'conservativity':
        return bool(result)
    
    return result

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/solve', methods=['GET', 'POST'])
def solve():
    if request.method == 'POST':
        operation = request.form.get('operation')
        if not operation:
            return redirect(url_for('home'))
        
        backend_key = OPERATION_MAP.get(operation, {}).get('key')
        if not backend_key:
            return redirect(url_for('home'))
        
        return render_template('solve.html', operation=operation)
    
    # Handle GET requests (when coming back from results)
    operation = request.args.get('operation')
    if operation in OPERATION_MAP:
        return render_template('solve.html', operation=operation)
    return redirect(url_for('home'))

@app.route('/result', methods=['POST'])
def result():
    operation = request.form.get('operation')
    problem_text = request.form.get('problem-statement', '').strip()
    
    logger.info(f"Solving {operation} problem: {problem_text[:50]}...")
    
    if not operation or not problem_text:
        error_msg = "Missing required fields. Please provide both operation type and problem statement."
        logger.warning(error_msg)
        return render_template('result.html', 
                            operation=operation,
                            error=error_msg)
    
    # Get backend operation key
    op_config = OPERATION_MAP.get(operation)
    if not op_config:
        error_msg = f"Invalid operation: {operation}"
        logger.warning(error_msg)
        return render_template('result.html',
                            operation=operation,
                            error=error_msg)
    
    backend_key = op_config['key']
    problems = PROBLEMS.get(backend_key, [])
    
    if not problems:
        error_msg = f"No problems available for {operation}"
        logger.warning(error_msg)
        return render_template('result.html',
                            operation=operation,
                            error=error_msg)
    
    # Find matching problem
    matched_problem = find_best_match(problem_text, problems, backend_key)
    if not matched_problem:
        error_msg = (f"No matching problem found for: '{problem_text}'. "
                    "Try being more specific about the vector field and curve/surface.")
        logger.warning(error_msg)
        return render_template('result.html',
                            operation=operation,
                            error=error_msg)
    
    logger.info(f"Matched problem: {matched_problem.get('id', 'unnamed')}")
    
    # Solve the problem
    solver_class = get_solver_class(backend_key)
    if not solver_class:
        error_msg = f"No solver available for {operation}"
        logger.error(error_msg)
        return render_template('result.html',
                            operation=operation,
                            error=error_msg)
    
    try:
        solver = solver_class()
        solution = None
        
        # Handle different operation types
        if backend_key == 'line_integral':
            solution = solver.solve_problem(matched_problem)
            if not solution['success']:
                raise ValueError(solution['error'])
            result_value = solution['result']
        elif backend_key == 'flux':
            result_value = solver.compute_flux(matched_problem)
        elif backend_key == 'flow':
            result_value = solver.compute_flow(matched_problem)
        elif backend_key == 'circulation':
            result_value = solver.compute_circulation(matched_problem)
        elif backend_key == 'work_integral':
            result_value = solver.compute_work(matched_problem)
        elif backend_key == 'conservativity':
            result_value = solver.is_conservative(
                matched_problem['function']['i'],
                matched_problem['function']['j'],
                matched_problem['function'].get('k', '0')
            )
        elif backend_key == 'potential_function':
            result_value = solver.find_potential_function(
                matched_problem['vector_field']['i'],
                matched_problem['vector_field']['j'],
                matched_problem['vector_field'].get('k', '0')
            )
        
        # Get exact solution if available
        exact_solution = matched_problem.get('solution', matched_problem.get('answer', 'N/A'))
        
        # Format numeric results
        if isinstance(result_value, (int, float)):
            result_value = round(result_value, 6)
        
        return render_template('result.html',
                    operation=operation,
                    result=result_value,
                    problem=matched_problem)
        
    except Exception as e:
        error_msg = f"Error solving problem: {str(e)}"
        logger.exception(error_msg)
        
        # Try to get partial results if available
        computed_result = None
        if 'solution' in locals() and isinstance(solution, dict) and 'result' in solution:
            computed_result = solution['result']
        
        return render_template('result.html',
                            operation=operation,
                            error=error_msg,
                            computed_result=computed_result)
        
        
if __name__ == '__main__':
    app.run(debug=True)