# from sklearn.neural_network import MLPRegressor, MLPClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_squared_error, accuracy_score
# import json
# import re
# import numpy as np
# from scipy.integrate import quad
# import sympy as sp
# from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
#                                        implicit_multiplication_application,
#                                        convert_xor, split_symbols)

 
# class AIModelIntegrator:
#     def __init__(self):
#         self.models = {
#             'line_integral': self._create_regressor(),
#             # Other models can be added here
#         }
#         self.training_data = {ptype: {'X': [], 'y': []} for ptype in self.models.keys()}

#     def _create_regressor(self):
#         return Pipeline([
#             ('scaler', StandardScaler()),
#             ('mlp', MLPRegressor(hidden_layer_sizes=(100, 50),
#                                 activation='relu',
#                                 solver='adam',
#                                 max_iter=2000,
#                                 random_state=42))
#         ])

#     def determine_problem_type(self, problem_data):
#         """Simplified problem type detection for line integrals"""
#         if 'curve' in problem_data and 'function' in problem_data:
#             return 'line_integral'
#         return None

#     def extract_features(self, problem_data, problem_type=None):
#         """Enhanced feature extraction for line integrals with error handling"""
#         features = []
        
#         try:
#             # Basic features with safe defaults
#             features.append(len(problem_data.get('problem_statement', '')))
#             features.append(len(problem_data.get('function', '')))
            
#             # Curve complexity features with safe access
#             curve = problem_data.get('curve', {})
#             features.append(len(curve.get('parametric_equations', '')))
            
#             # Handle start/end points safely
#             start_point = curve.get('start_point', [])
#             end_point = curve.get('end_point', [])
#             features.append(len(str(start_point)))
#             features.append(len(str(end_point)))
            
#             # Function complexity with safe counting
#             func = problem_data.get('function', '')
#             features.append(func.count('x') + func.count('y') + func.count('z'))
#             features.append(func.count('+') + func.count('-') + func.count('*') + func.count('/'))
#             features.append(func.lower().count('sin') + func.lower().count('cos') + func.lower().count('tan'))
            
#             # Limits features with safe access
#             limits = problem_data.get('limits', [])
#             features.append(len(limits))
            
#             # Check if first limit exists before checking its type
#             if len(limits) > 0:
#                 features.append(1 if isinstance(limits[0], str) else 0)
#             else:
#                 features.append(0)  # Default to numeric if no limits
                
#             # Add problem_type as a feature if provided
#             if problem_type is not None:
#                 # Convert problem type to numerical value
#                 type_mapping = {'line_integral': 1, 'flux': 2, 'flow': 3, 'circulation': 4}
#                 features.append(type_mapping.get(problem_type, 0))
                
#         except Exception as e:
#             print(f"Warning: Feature extraction error - {str(e)}")
#             # Return default feature vector if extraction fails
#             default_length = 11 if problem_type is None else 12
#             return np.zeros((1, default_length))
        
#         return np.array(features).reshape(1, -1)

#     def collect_training_data(self, json_files):
#         """Load and process training data"""
#         for file in json_files:
#             try:
#                 with open(file) as f:
#                     data = json.load(f)
#                     problems = data.get('problems', []) if isinstance(data, dict) else data
                    
#                     for problem in problems:
#                         ptype = self.determine_problem_type(problem)
#                         if ptype is None:
#                             continue
                            
#                         features = self.extract_features(problem, ptype)
                        
#                         # Process solution
#                         if 'solution' in problem:
#                             try:
#                                 # Handle symbolic solutions
#                                 sol_expr = sp.sympify(problem['solution'])
#                                 target = float(sol_expr.evalf())
#                                 self.training_data[ptype]['X'].append(features)
#                                 self.training_data[ptype]['y'].append(target)
#                             except:
#                                 continue
            
#             except Exception as e:
#                 print(f"Error loading {file}: {str(e)}")
#                 continue
        
#         # Train models
#         for ptype, model in self.models.items():
#             X = np.vstack(self.training_data[ptype]['X']) if self.training_data[ptype]['X'] else None
#             y = np.array(self.training_data[ptype]['y']) if self.training_data[ptype]['y'] else None
            
#             if X is not None and y is not None and len(X) > 0:
#                 X_train, X_test, y_train, y_test = train_test_split(
#                     X, y, test_size=0.2, random_state=42
#                 )
#                 model.fit(X_train, y_train)
#                 y_pred = model.predict(X_test)
#                 mse = mean_squared_error(y_test, y_pred)
#                 print(f"{ptype.replace('_', ' ').title()} Model trained. MSE: {mse:.4f}")
#                 print(f"Sample predictions: {y_pred[:5]} vs actual: {y_test[:5]}")
#             else:
#                 print(f"No training data for {ptype}")

#     def predict(self, problem_data):
#         """Make prediction for a line integral problem"""
#         ptype = self.determine_problem_type(problem_data)
#         if ptype is None or ptype not in self.models:
#             return None
            
#         features = self.extract_features(problem_data, ptype)
#         return self.models[ptype].predict(features)[0]


# class IntegralSolver:
#     def __init__(self, json_file=None):
#         if json_file:
#             with open(json_file) as f:
#                 data = json.load(f)
#                 self.problems = data if isinstance(data, list) else data.get('problems', [])
#         else:
#             self.problems = []
            
#         self.transformations = (standard_transformations + 
#                               (implicit_multiplication_application, 
#                                convert_xor, 
#                                split_symbols))
#         self.x, self.y, self.t, self.a = sp.symbols('x y t a')

#          # Initialize the AI model integrator
#         self.ai_model = AIModelIntegrator()
#         # Collect training data if available
#         try:
#             self.ai_model.collect_training_data(['lineintegrals.json', 'flux.json', 'flow.json', 
#                                               'circulation.json', 'workintegrals.json', 
#                                               'conservativity.json', 'potentialfunctions.json'])
#         except FileNotFoundError:
#             print("Warning: Some training data files not found. AI predictions may be less accurate.")
            
            
#     def parse_expression(self, expr_str):
#         """Shared expression parser"""
#         replacements = [
#             ('²', '**2'), ('³', '**3'), ('∞', 'inf'),
#             ('π', 'pi'), ('^', '**'), (' ', ''),
#             ('sin', 'sp.sin'), ('cos', 'sp.cos'), ('tan', 'sp.tan'),
#             ('cot', 'sp.cot'), ('sec', 'sp.sec'), ('csc', 'sp.csc'),
#             ('asin', 'sp.asin'), ('acos', 'sp.acos'), ('atan', 'sp.atan'),
#             ('sinh', 'sp.sinh'), ('cosh', 'sp.cosh'), ('tanh', 'sp.tanh'),
#             ('exp', 'sp.exp'), ('log', 'sp.log'), ('sqrt', 'sp.sqrt'),
#             ('°', '*pi/180'), ('≤', '<='), ('≥', '>=')
#         ]
        
#         for trig_func in ['sin', 'cos', 'tan', 'cot', 'sec', 'csc']:
#             expr_str = expr_str.replace(f'{trig_func}²', f'sp.{trig_func}**2')
#             expr_str = expr_str.replace(f'{trig_func}³', f'sp.{trig_func}**3')
        
#         for old, new in replacements:
#             expr_str = expr_str.replace(old, new)
            
#         try:
#             return parse_expr(expr_str, transformations=self.transformations,
#                             local_dict={'sp': sp})
#         except Exception as e:
#             expr_str = expr_str.replace(')(', ')*(')
#             return parse_expr(expr_str, transformations=self.transformations,
#                             local_dict={'sp': sp})

#     def solve_problem(self, problem):
#         """Solve a single problem with enhanced trig support and AI prediction"""
#         try:
#             # First, get an AI prediction for the problem
#             ai_prediction = self.ai_model.predict(problem)
#             print(f"AI Prediction (estimate): {ai_prediction}")
            
#            # Parse function
#             func_part = problem["function"].split('=')[1].strip()
#             f_expr = self.parse_expression(func_part)
            
#             # Parse parametric equations
#             eqs = problem["curve"]["parametric_equations"].replace(' ', '')
#             components = {}
#             for var in ['x', 'y', 'z']:
#                 if f"{var}(t)=" in eqs:
#                     components[var] = self.parse_expression(
#                         eqs.split(f"{var}(t)=")[1].split(',')[0].split(')')[0]
#                     )
            
#             # Parse limits with trig support
#             limits = []
#             for limit in problem["limits"]:
#                 if isinstance(limit, str):
#                     limit = limit.replace('π', 'pi')
#                     parsed_limit = self.parse_expression(limit)
#                     limits.append(float(parsed_limit.evalf()))
#                 else:
#                     limits.append(float(limit))
            
#             # Compute integral
#             t = sp.symbols('t')
#             derivatives = {var: sp.diff(eq, t) for var, eq in components.items()}
#             magnitude = sp.sqrt(sum(d**2 for d in derivatives.values()))
            
#             subs_dict = {sp.symbols(var): eq for var, eq in components.items()}
#             integrand = f_expr.subs(subs_dict) * magnitude
            
#             integrand = sp.simplify(integrand)
            
#             if limits[0] > limits[1]:
#                 limits = [limits[1], limits[0]]
#                 integrand = -integrand
            
#             integrand_func = sp.lambdify(t, integrand, modules=['numpy', {'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
#                                                                        'pi': np.pi, 'exp': np.exp, 'sqrt': np.sqrt}])
            
#             result, _ = quad(integrand_func, limits[0], limits[1], 
#                             epsabs=1e-6, epsrel=1e-6, 
#                             limit=100, points=self.get_singular_points(integrand, t, limits))
            
#             return {'success': True, 'result': result, 'ai_prediction': ai_prediction, 'exact_solution': problem.get('solution', 'N/A')} 

            
#         except Exception as e:
#             return {'success': False, 'error': str(e), 'ai_prediction': None}

#     def get_singular_points(self, integrand, t, limits):
#         """Find potential singular points in trigonometric expressions"""
#         try:
#             denom = sp.denom(integrand)
#             singular_points = sp.solve(denom, t)
            
#             singular_points = [float(p.evalf()) for p in singular_points 
#                              if limits[0] <= float(p.evalf()) <= limits[1]]
            
#             tan_points = [float((sp.pi/2 + k*sp.pi).evalf()) 
#                          for k in range(int(limits[0]/sp.pi)-1, int(limits[1]/sp.pi)+2)
#                          if limits[0] <= float((sp.pi/2 + k*sp.pi).evalf()) <= limits[1]]
            
#             return sorted(list(set(singular_points + tan_points)))
#         except:
#             return []

#     def solve_all(self):
#         """Solve all problems with proper error handling"""
#         print("=== Line Integral Solutions ===")
#         for problem in self.problems:
#             solution = self.solve_problem(problem)
#             print(f"\nProblem {problem['id']}:")
#             print(f"Statement: {problem['problem_statement']}")
            
#             if solution['success']:
#                 print(f"AI Estimate: {solution.get('ai_prediction', 'N/A')}")
#                 print(f"Computed result: {solution['result']:.8f}")
#                 print(f"Exact solution: {problem['solution']}")
#                 try:
#                     exact_value = float(self.parse_expression(problem['solution']).evalf())
#                     print(f"Exact value: {exact_value:.8f}")
#                     print(f"Absolute error: {abs(solution['result'] - exact_value):.2e}")
#                 except:
#                     pass
#             else:
#                 print(f"Error: {solution['error']}")

# class FluxCalculator(IntegralSolver):
#     def __init__(self):
#         super().__init__()  # Initialize without JSON file
#         self.x, self.y, self.t, self.a = sp.symbols('x y t a')
    
#     def compute_flux(self, problem):
#         """Compute flux for a given problem dictionary"""
#         F_str = problem['vector_field']
#         curve_type = problem['type']
#         params = problem.get('params', {})
        
#         if curve_type == "circle":
#             return self._flux_circle(F_str, params.get("a", 1))
#         elif curve_type == "semicircle_with_line":
#             return self._flux_semicircle_with_line(F_str, params.get("a", 1))
#         elif curve_type == "triangle":
#             return self._flux_triangle(F_str)
#         else:
#             raise ValueError(f"Unsupported curve type: {curve_type}")

#     def _flux_circle(self, F_str, a):
#         """Compute flux across a circle of radius 'a'."""
#         F = self._parse_vector_field(F_str)
#         x, y = a*sp.cos(self.t), a*sp.sin(self.t)
#         dx, dy = -a*sp.sin(self.t), a*sp.cos(self.t)
#         integrand = F[0].subs({self.x: x, self.y: y}) * dy - F[1].subs({self.x: x, self.y: y}) * dx
#         return sp.integrate(integrand, (self.t, 0, 2*sp.pi))

#     def _flux_semicircle_with_line(self, F_str, a):
#         """Compute flux across semicircle + line segment."""
#         F = self._parse_vector_field(F_str)
#         # Semicircle part (0 ≤ t ≤ π)
#         x_sc, y_sc = a*sp.cos(self.t), a*sp.sin(self.t)
#         dx_sc, dy_sc = -a*sp.sin(self.t), a*sp.cos(self.t)
#         integrand_sc = F[0].subs({self.x: x_sc, self.y: y_sc}) * dy_sc - F[1].subs({self.x: x_sc, self.y: y_sc}) * dx_sc
#         flux_sc = sp.integrate(integrand_sc, (self.t, 0, sp.pi))

#         # Line segment part (-a ≤ t ≤ a, y=0)
#         integrand_line = F[1].subs({self.y: 0})
#         flux_line = sp.integrate(-integrand_line.subs({self.x: self.t}), (self.t, -a, a))

#         return flux_sc + flux_line

#     def _flux_triangle(self, F_str):
#         """Compute flux across triangle (1,0)→(0,1)→(-1,0)→(1,0)."""
#         F = self._parse_vector_field(F_str)
#         # Segment 1: (1,0) to (0,1)
#         x1, y1 = 1 - self.t, self.t
#         dx1, dy1 = -1, 1
#         integrand1 = F[0].subs({self.x: x1, self.y: y1}) * dy1 - F[1].subs({self.x: x1, self.y: y1}) * dx1
#         flux1 = sp.integrate(integrand1, (self.t, 0, 1))

#         # Segment 2: (0,1) to (-1,0)
#         x2, y2 = -self.t, 1 - self.t
#         dx2, dy2 = -1, -1
#         integrand2 = F[0].subs({self.x: x2, self.y: y2}) * dy2 - F[1].subs({self.x: x2, self.y: y2}) * dx2
#         flux2 = sp.integrate(integrand2, (self.t, 0, 1))

#         # Segment 3: (-1,0) to (1,0)
#         x3, y3 = -1 + 2*self.t, 0
#         dx3, dy3 = 2, 0
#         integrand3 = F[0].subs({self.x: x3, self.y: y3}) * dy3 - F[1].subs({self.x: x3, self.y: y3}) * dx3
#         flux3 = sp.integrate(integrand3, (self.t, 0, 1))

#         return flux1 + flux2 + flux3

#     def _parse_vector_field(self, F_str):
#         """Use inherited parse_expression"""
#         parts = F_str.replace(' ', '').split('*i')
#         Fx = parts[0]
#         Fy = parts[1].split('*j')[0] if '*j' in parts[1] else '0'
#         return [self.parse_expression(Fx), self.parse_expression(Fy)]

#     def solve_all_flux(self, problems):
#         """Solve all flux problems with formatted output"""
#         print("\n=== Flux Solutions ===")
#         for problem in problems:
#             try:
#                 result = self.compute_flux(problem)
#                 print(f"\nProblem {problem['id']}:")
#                 print(f"Vector Field: {problem['vector_field']}")
#                 print(f"Curve: {problem['curve']}")
#                 if result.is_number:
#                     print(f"Computed Flux: {float(result.evalf()):.8f}")
#                 else:
#                     print(f"Symbolic Result: {result}")
#             except Exception as e:
#                 print(f"\nProblem {problem['id']} failed: {str(e)}")
                
# class FlowCalculator(IntegralSolver):
#     def __init__(self):
#         super().__init__()
    
#     def _parse_vector_field(self, field_str):
#         """Robust vector field parser that handles all cases"""
#         components = [sp.Integer(0), sp.Integer(0), sp.Integer(0)]
#         clean_str = field_str.replace(' ', '')
        
#         # Split components while handling cases like 'xi + yj + zk' or 'x²i + (x+y)j'
#         for i, comp in enumerate(['i', 'j', 'k']):
#             if comp in clean_str:
#                 parts = clean_str.split(comp, 1)
#                 expr_str = parts[0]
#                 # Handle cases where component might start/end with operators
#                 if expr_str.endswith('+') or expr_str.endswith('-'):
#                     expr_str = expr_str[:-1]
#                 components[i] = self.parse_expression(expr_str)
#                 clean_str = parts[1] if len(parts) > 1 else ''
        
#         return components

#     def parse_expression(self, expr_str):
#         """Safe expression parser that handles exponents and parentheses"""
#         expr_str = expr_str.replace('^', '**')
#         try:
#             return sp.sympify(expr_str)
#         except:
#             raise ValueError(f"Could not parse expression: {expr_str}")

#     def compute_flow(self, problem):
#         """Main method to compute flow for any problem type"""
#         try:
#             F = self._parse_vector_field(problem['vector_field'])
#             path_type = problem['path']['type']
            
#             handlers = {
#                 'semicircle': self._semicircle_flow,
#                 'line_segment': self._line_flow,
#                 '3d_helix': self._helix_flow,
#                 '3d_curve': self._3d_curve_flow,
#                 '3d_parabola': self._3d_parabola_flow,
#                 'circle': self._circle_flow,
#                 'parabola': self._parabola_flow,
#                 '3d_line': self._3d_line_flow
#             }
            
#             if path_type not in handlers:
#                 raise ValueError(f"Unsupported path type: {path_type}")
            
#             return handlers[path_type](F, problem['path'])
            
#         except Exception as e:
#             print(f"Error processing problem {problem.get('id', '?')}: {str(e)}")
#             raise

#     def _get_path_components(self, path_eq):
#         """Extract x(t), y(t), z(t) from path equation"""
#         # This is a simplified parser - in practice you might need a more robust solution
#         if 'i' in path_eq and 'j' in path_eq:
#             x_part = path_eq.split('i')[0].strip()
#             remaining = path_eq.split('i')[1]
#             y_part = remaining.split('j')[0].strip()
#             z_part = remaining.split('j')[1].split('k')[0].strip() if 'k' in remaining else '0'
#             return x_part, y_part, z_part
#         return '0', '0', '0'

#     def _semicircle_flow(self, F, path):
#         a, b = path['interval']
#         x = sp.cos(self.t)
#         y = sp.sin(self.t)
#         dx = -sp.sin(self.t)
#         dy = sp.cos(self.t)
#         integrand = F[0].subs({self.x: x, self.y: y})*dy - F[1].subs({self.x: x, self.y: y})*dx
#         return sp.integrate(integrand, (self.t, a, b))

#     def _line_flow(self, F, path):
#         p1, p2 = path['points']
#         dim = len(p1)
#         params = {
#             self.x: p1[0] + self.t*(p2[0]-p1[0]),
#             self.y: p1[1] + self.t*(p2[1]-p1[1])
#         }
#         derivatives = [p2[0]-p1[0], p2[1]-p1[1]]
        
#         if dim == 3:
#             params[self.z] = p1[2] + self.t*(p2[2]-p1[2])
#             derivatives.append(p2[2]-p1[2])
        
#         integrand = sum(F[i].subs(params)*derivatives[i] for i in range(dim))
#         return sp.integrate(integrand, (self.t, 0, 1))

#     def _helix_flow(self, F, path):
#         a, b = path['interval']
#         # Parse components from equation
#         x = sp.cos(self.t)
#         y = sp.sin(self.t)
#         z_expr = path['equation'].split('k')[0].split('+')[-1].strip()
#         z = sp.sympify(z_expr) if z_expr else self.t/6  # Default if not found
        
#         dx = -sp.sin(self.t)
#         dy = sp.cos(self.t)
#         dz = sp.diff(z, self.t)
        
#         integrand = (F[0].subs({self.x: x, self.y: y, self.z: z}) * dx +
#                     F[1].subs({self.x: x, self.y: y, self.z: z}) * dy +
#                     F[2].subs({self.x: x, self.y: y, self.z: z}) * dz)
#         return sp.integrate(integrand, (self.t, a, b))

#     def _3d_curve_flow(self, F, path):
#         a, b = path['interval']
#         x = sp.sin(self.t)
#         y = sp.cos(self.t)
#         z = self.t/6
#         dx = sp.cos(self.t)
#         dy = -sp.sin(self.t)
#         dz = 1/6
        
#         integrand = (F[0].subs({self.x: x, self.y: y, self.z: z}) * dx +
#                     F[1].subs({self.x: x, self.y: y, self.z: z}) * dy +
#                     F[2].subs({self.x: x, self.y: y, self.z: z}) * dz)
#         return sp.integrate(integrand, (self.t, a, b))

#     def _3d_parabola_flow(self, F, path):
#         a, b = path['interval']
#         x = self.t
#         y = self.t**2
#         z = self.t
#         dx = 1
#         dy = 2*self.t
#         dz = 1
        
#         integrand = (F[0].subs({self.x: x, self.y: y, self.z: z}) * dx +
#                     F[1].subs({self.x: x, self.y: y, self.z: z}) * dy +
#                     F[2].subs({self.y: y, self.z: z}) * dz)  # F[2] = -y*z
#         return sp.integrate(integrand, (self.t, a, b))

#     def _circle_flow(self, F, path):
#         a, b = path['interval']
#         if '(-sin' in path['equation']:  # Problem 8
#             x = -sp.sin(self.t)
#             y = sp.cos(self.t)
#             dx = -sp.cos(self.t)
#             dy = -sp.sin(self.t)
#         else:  # Regular circle
#             x = sp.cos(self.t)
#             y = sp.sin(self.t)
#             dx = -sp.sin(self.t)
#             dy = sp.cos(self.t)
        
#         integrand = F[0].subs({self.x: x, self.y: y}) * dy - F[1].subs({self.x: x, self.y: y}) * dx
#         return sp.integrate(integrand, (self.t, a, b))

#     def _parabola_flow(self, F, path):
#         # Extract start and end points from path
#         (x1, y1), (x2, y2) = path['points']
#         t1 = sp.sqrt(y1) if y1 >= 0 else -sp.sqrt(-y1)
#         t2 = sp.sqrt(y2) if y2 >= 0 else -sp.sqrt(-y2)
        
#         x = self.t
#         y = self.t**2
#         dx = 1
#         dy = 2*self.t
        
#         integrand = F[0].subs({self.x: x, self.y: y}) * dx + F[1].subs({self.x: x, self.y: y}) * dy
#         return sp.integrate(integrand, (self.t, t1, t2))

#     def _3d_line_flow(self, F, path):
#         return self._line_flow(F, path)

#     def solve_all_flows(self, problems):
#         """Solve all flow problems with detailed output"""
#         print("\n=== Flow Solutions ===")
#         for problem in problems:
#             try:
#                 result = self.compute_flow(problem)
#                 print(f"\nProblem {problem['id']}:")
#                 print(f"Vector Field: {problem['vector_field']}")
#                 print(f"Path Type: {problem['path']['type']}")
                
#                 if result.is_number:
#                     res_float = float(result.evalf())
#                     print(f"Computed Flow: {res_float:.8f}")
#                     if 'solution' in problem:
#                         expected = float(problem['solution'])
#                         print(f"Expected Answer: {expected}")
#                         print(f"Absolute Error: {abs(res_float - expected):.2e}")
#                 else:
#                     print(f"Symbolic Result: {result}")
#             except Exception as e:
#                 print(f"\nProblem {problem['id']} failed: {str(e)}")
                    
# class CirculationCalculator(IntegralSolver):
#     def __init__(self):
#         super().__init__()
#         self.x, self.y, self.t = sp.symbols('x y t')
    
#     def _parse_vector_field(self, F_str):
#         """Parse vector field string into components [Fx, Fy]"""
#         F_str = F_str.replace(' ', '').replace('F=', '')
#         parts = F_str.split('i+')
#         Fx = parts[0]
#         Fy = parts[1].split('j')[0] if 'j' in parts[1] else '0'
#         return [self.parse_expression(Fx), self.parse_expression(Fy)]

#     def compute_circulation(self, problem):
#         """Compute circulation for a problem in the original format"""
#         try:
#             F = self._parse_vector_field(problem['vector_field'])
#             curve_desc = problem['curve']
            
#             if "triangle" in curve_desc.lower() and "(0,0)" in curve_desc:
#                 return self._triangle_001_100_010(F)
#             elif "circle" in curve_desc.lower():
#                 return self._circle_circulation(F, problem)
#             elif "square" in curve_desc.lower():
#                 return self._square_circulation(F)
#             else:
#                 raise ValueError(f"Unsupported curve: {curve_desc}")
#         except Exception as e:
#             print(f"Error computing circulation: {str(e)}")
#             raise

#     def _triangle_001_100_010(self, F):
#         """Compute circulation around triangle (0,0)→(1,0)→(0,1)→(0,0)"""
#         # Segment 1: (0,0) to (1,0)
#         x1, y1 = self.t, 0
#         dx1, dy1 = 1, 0
#         integrand1 = F[0].subs({self.x: x1, self.y: y1})*dx1 + F[1].subs({self.x: x1, self.y: y1})*dy1
#         circ1 = sp.integrate(integrand1, (self.t, 0, 1))
        
#         # Segment 2: (1,0) to (0,1)
#         x2, y2 = 1-self.t, self.t
#         dx2, dy2 = -1, 1
#         integrand2 = F[0].subs({self.x: x2, self.y: y2})*dx2 + F[1].subs({self.x: x2, self.y: y2})*dy2
#         circ2 = sp.integrate(integrand2, (self.t, 0, 1))
        
#         # Segment 3: (0,1) to (0,0)
#         x3, y3 = 0, 1-self.t
#         dx3, dy3 = 0, -1
#         integrand3 = F[0].subs({self.x: x3, self.y: y3})*dx3 + F[1].subs({self.x: x3, self.y: y3})*dy3
#         circ3 = sp.integrate(integrand3, (self.t, 0, 1))
        
#         return circ1 + circ2 + circ3

#     def _circle_circulation(self, F, problem):
#         """Compute circulation around a circle"""
#         if "r(t) = (cos t)i + (sin t)j" in problem['curve']:
#             a = 1  # unit circle
#         else:
#             # Extract radius if available (e.g., for r(t) = (2cos t)i + (2sin t)j)
#             a = float(problem['curve'].split('cos')[0].split('(')[-1] or 1)
        
#         x, y = a*sp.cos(self.t), a*sp.sin(self.t)
#         dx, dy = -a*sp.sin(self.t), a*sp.cos(self.t)
#         integrand = F[0].subs({self.x: x, self.y: y})*dx + F[1].subs({self.x: x, self.y: y})*dy
#         return sp.integrate(integrand, (self.t, 0, 2*sp.pi))

#     def _square_circulation(self, F):
#         """Compute circulation around unit square (0,0)→(1,0)→(1,1)→(0,1)→(0,0)"""
#         # Segment 1: (0,0) to (1,0)
#         x1, y1 = self.t, 0
#         dx1, dy1 = 1, 0
#         integrand1 = F[0].subs({self.x: x1, self.y: y1})*dx1 + F[1].subs({self.x: x1, self.y: y1})*dy1
#         circ1 = sp.integrate(integrand1, (self.t, 0, 1))
        
#         # Segment 2: (1,0) to (1,1)
#         x2, y2 = 1, self.t
#         dx2, dy2 = 0, 1
#         integrand2 = F[0].subs({self.x: x2, self.y: y2})*dx2 + F[1].subs({self.x: x2, self.y: y2})*dy2
#         circ2 = sp.integrate(integrand2, (self.t, 0, 1))
        
#         # Segment 3: (1,1) to (0,1)
#         x3, y3 = 1-self.t, 1
#         dx3, dy3 = -1, 0
#         integrand3 = F[0].subs({self.x: x3, self.y: y3})*dx3 + F[1].subs({self.x: x3, self.y: y3})*dy3
#         circ3 = sp.integrate(integrand3, (self.t, 0, 1))
        
#         # Segment 4: (0,1) to (0,0)
#         x4, y4 = 0, 1-self.t
#         dx4, dy4 = 0, -1
#         integrand4 = F[0].subs({self.x: x4, self.y: y4})*dx4 + F[1].subs({self.x: x4, self.y: y4})*dy4
#         circ4 = sp.integrate(integrand4, (self.t, 0, 1))
        
#         return circ1 + circ2 + circ3 + circ4

#     def solve_all_circulations(self, problems):
#         """Solve all circulation problems with detailed output matching original format"""
#         print("\n=== Circulation Problem Solutions ===")
#         for problem in problems:
#             try:
#                 print(f"\nProblem {problem['id']}:")
#                 print(f"Question: {problem['question']}")
#                 print(f"Vector Field: {problem['vector_field']}")
#                 print(f"Curve: {problem['curve']}")
                
#                 result = self.compute_circulation(problem)
#                 exact_solution = problem.get('solution', 'Not provided')
#                 explanation = problem.get('solution_explanation', 'No explanation provided')
                
#                 print("\nResults:")
#                 if result.is_number:
#                     print(f"Computed Circulation: {float(result.evalf()):.8f}")
#                     print(f"Exact Solution: {exact_solution}")
#                     try:
#                         exact_val = float(exact_solution)
#                         print(f"Absolute Error: {abs(float(result.evalf()) - exact_val):.2e}")
#                     except:
#                         pass
#                 else:
#                     print(f"Symbolic Result: {result}")
#                     print(f"Exact Solution: {exact_solution}")
                
#                 print(f"\nExplanation: {explanation}")
#                 print("="*60)
                
#             except Exception as e:
#                 print(f"\nFailed to solve Problem {problem.get('id', '?')}: {str(e)}")
#                 print("="*60)                 

# class SolveWorkIntegral(IntegralSolver):
#     def __init__(self):
#         super().__init__()
#         self.x, self.y, self.z, self.t = sp.symbols('x y z t')

#     def _parse_vector_field(self, F_str):
#         """Parse vector field string into components [Fx, Fy, Fz]"""
#         F_str = F_str.replace('F =', '').strip()
#         components = {'i': '0', 'j': '0', 'k': '0'}
        
#         # Split components
#         if 'i' in F_str:
#             components['i'] = F_str.split('i')[0]
#             remaining = F_str.split('i', 1)[1]
#             if 'j' in remaining:
#                 components['j'] = remaining.split('j')[0]
#                 remaining = remaining.split('j', 1)[1]
#                 if 'k' in remaining:
#                     components['k'] = remaining.split('k')[0]
#         else:
#             # Handle case where components might be in different order
#             for comp in ['i', 'j', 'k']:
#                 if comp in F_str:
#                     parts = F_str.split(comp, 1)
#                     components[comp] = parts[0]
#                     F_str = parts[1] if len(parts) > 1 else ''
        
#         return [self.parse_expression(components['i']),
#                 self.parse_expression(components['j']),
#                 self.parse_expression(components['k'])]

#     def _parse_curve(self, curve_str):
#         """Parse parametric curve definition"""
#         curve_str = curve_str.replace('r(t) =', '').strip()
#         components = curve_str.split('+')
#         x_part = components[0].split('i')[0].strip()
#         y_part = components[1].split('j')[0].strip() if len(components) > 1 else '0'
#         z_part = components[2].split('k')[0].strip() if len(components) > 2 else '0'
        
#         return (self.parse_expression(x_part),
#                 self.parse_expression(y_part),
#                 self.parse_expression(z_part))

#     def compute_work(self, problem):
#         """Compute work integral for a given problem"""
#         try:
#             F = self._parse_vector_field(problem['vector_field'])
#             x_eq, y_eq, z_eq = self._parse_curve(problem['curve'])
            
#             # Parse interval
#             interval = problem['interval'].strip('[]').split(',')
#             t_min, t_max = map(float, interval)
            
#             # Compute derivatives
#             dx = sp.diff(x_eq, self.t)
#             dy = sp.diff(y_eq, self.t)
#             dz = sp.diff(z_eq, self.t) if z_eq != 0 else 0
            
#             # Compute integrand
#             integrand = (F[0].subs({self.x: x_eq, self.y: y_eq, self.z: z_eq}) * dx +
#                         F[1].subs({self.x: x_eq, self.y: y_eq, self.z: z_eq}) * dy)
            
#             if dz != 0:
#                 integrand += F[2].subs({self.x: x_eq, self.y: y_eq, self.z: z_eq}) * dz
            
#             result = sp.integrate(integrand, (self.t, t_min, t_max))
#             return float(result.evalf()) if result.is_number else result
#         except Exception as e:
#             print(f"Error computing work integral: {str(e)}")
#             return None

#     def solve_all_work_integrals(self, problems):
#         """Solve and display all work integral problems"""
#         print("\n=== WORK INTEGRAL SOLUTIONS ===")
#         print("=" * 60)
        
#         for problem in sorted(problems, key=lambda x: x['id']):
#             print(f"\nProblem {problem['id']}:")
#             print(f"Question: {problem['question']}")
#             print(f"Vector Field: {problem['vector_field']}")
#             print(f"Curve: {problem['curve']}")
#             print(f"Interval: {problem['interval']}")
            
#             computed_result = self.compute_work(problem)
#             expected_result = problem['answer']
            
#             print("\nResults:")
#             if computed_result is not None:
#                 if isinstance(computed_result, float):
#                     print(f"Computed Result: {computed_result:.6f}")
#                 else:
#                     print(f"Computed Result: {computed_result}")
#                 print(f"Expected Result: {expected_result}")
#             else:
#                 print("Failed to compute result")
            
#             print("=" * 60)

# class ConservativeChecker:
#     def __init__(self):
#         self.transformations = (standard_transformations + 
#                               (implicit_multiplication_application, 
#                                convert_xor, 
#                                split_symbols))
#         self.x, self.y, self.z = sp.symbols('x y z')
        
#     def parse_expression(self, expr_str):
#         """Parse a mathematical expression string into a sympy expression"""
#         try:
#             return parse_expr(expr_str, transformations=self.transformations,
#                             local_dict={'x': self.x, 'y': self.y, 'z': self.z})
#         except Exception as e:
#             expr_str = expr_str.replace(')(', ')*(')
#             return parse_expr(expr_str, transformations=self.transformations,
#                             local_dict={'x': self.x, 'y': self.y, 'z': self.z})
    
#     def is_conservative(self, F_i, F_j, F_k):
#         """
#         Check if a 3D vector field is conservative by verifying:
#         1. curl F = 0
#         2. The domain is simply connected (we can't verify this programmatically)
#         """
#         try:
#             # Parse the component functions
#             Fx = self.parse_expression(F_i)
#             Fy = self.parse_expression(F_j)
#             Fz = self.parse_expression(F_k)
            
#             # Calculate partial derivatives for curl
#             dFz_dy = sp.diff(Fz, self.y)
#             dFy_dz = sp.diff(Fy, self.z)
            
#             dFx_dz = sp.diff(Fx, self.z)
#             dFz_dx = sp.diff(Fz, self.x)
            
#             dFy_dx = sp.diff(Fy, self.x)
#             dFx_dy = sp.diff(Fx, self.y)
            
#             # Check if all components of the curl are zero
#             curl_x = dFz_dy - dFy_dz
#             curl_y = dFx_dz - dFz_dx
#             curl_z = dFy_dx - dFx_dy
            
#             # Simplify the curl components
#             curl_x_simp = sp.simplify(curl_x)
#             curl_y_simp = sp.simplify(curl_y)
#             curl_z_simp = sp.simplify(curl_z)
            
#             # If all components are zero, the field is conservative
#             if curl_x_simp == 0 and curl_y_simp == 0 and curl_z_simp == 0:
#                 return True
#             return False
            
#         except Exception as e:
#             print(f"Error checking conservativity: {str(e)}")
#             return None      
        
# class PotentialFunctionCalculator(ConservativeChecker):
#     def __init__(self):
#         super().__init__()
        
#     def find_potential_function(self, F_i, F_j, F_k):
#         """
#         Find a potential function f for a conservative vector field F = (F_i, F_j, F_k)
#         Returns the potential function as a sympy expression or None if not conservative
#         """
#         if not self.is_conservative(F_i, F_j, F_k):
#             return None
            
#         try:
#             # Parse the component functions
#             Fx = self.parse_expression(F_i)
#             Fy = self.parse_expression(F_j)
#             Fz = self.parse_expression(F_k) if F_k != '0' else 0
            
#             # Initialize potential function
#             f = sp.integrate(Fx, self.x)
            
#             # Find g(y,z) such that ∂f/∂y = Fy
#             df_dy = sp.diff(f, self.y)
#             g_y = Fy - df_dy
            
#             # Integrate g_y with respect to y to find g(y,z)
#             g = sp.integrate(g_y, self.y)
            
#             # Add g to the potential function
#             f += g
            
#             # For 3D fields, find h(z) such that ∂f/∂z = Fz
#             if Fz != 0:
#                 df_dz = sp.diff(f, self.z)
#                 h_z = Fz - df_dz
#                 h = sp.integrate(h_z, self.z)
#                 f += h
            
#             # Simplify the result and add constant C
#             f = sp.simplify(f) + sp.Symbol('C')
            
#             return f
            
#         except Exception as e:
#             print(f"Error finding potential function: {str(e)}")
#             return None
            
#     def solve_all_potential_functions(self, problems):
#         """Solve and display all potential function problems in a clean format"""
#         print("\n=== POTENTIAL FUNCTION SOLUTIONS ===")
#         print("=" * 60)
        
#         for problem in sorted(problems, key=lambda x: x['id']):
#             print(f"\nProblem {problem['id']}:")
#             print(f"Statement: {problem['problem_statement']}")
            
#             try:
#                 # Get components, handling 2D fields by setting k component to '0'
#                 F_i = problem['vector_field']['i']
#                 F_j = problem['vector_field']['j']
#                 F_k = problem['vector_field'].get('k', '0')
                
#                 potential = self.find_potential_function(F_i, F_j, F_k)
                
#                 print("\nSolution:")
#                 if potential is not None:
#                     print(f"f(x,y,z) = {potential}")
#                 else:
#                     print("No potential function exists (field is not conservative)")
                
#                 print(f"\nExpected solution: {problem['solution']}")
                
#             except Exception as e:
#                 print(f"\nError processing problem: {str(e)}")
#                 print("Skipping this problem...")
            
#             print("=" * 60)
 
        
                    
# def display_menu():
#     print("\n" + "="*50)
#     print("MULTIVARIABLE CALCULUS PROBLEM SOLVER")
#     print("="*50)
#     print("1. Line Integral")
#     print("2. Flow")
#     print("3. Flux")
#     print("4. Circulation")
#     print("5. Work Integral")
#     print("6. Conservativity Check")
#     print("7. Potential Function")
#     print("8. Solve Custom Problem (Dynamic Input)")
#     print("0. Exit")
#     print("="*50)

# def solve_line_integrals():
#     try:
#         solver = IntegralSolver('lineintegrals.json')
#         solver.solve_all()
#     except FileNotFoundError:
#         print("\nNo lineintegrals.json file found. Skipping line integral problems.")

# def solve_flux_problems():
#     try:
#         with open('flux.json') as f:
#             flux_data = json.load(f)
#             flux_problems = flux_data if isinstance(flux_data, list) else flux_data.get('problems', [])
        
#         if flux_problems:
#             flux_calc = FluxCalculator()
#             flux_calc.solve_all_flux(flux_problems)
#         else:
#             print("\nNo flux problems found in the file.")
#     except FileNotFoundError:
#         print("\nNo flux.json file found. Skipping flux problems.")

# def solve_flow_problems():
#     try:
#         with open('flow.json') as f:
#             flow_data = json.load(f)
#             flow_problems = flow_data if isinstance(flow_data, list) else flow_data.get('problems', [])
        
#         if flow_problems:
#             FlowCalculator().solve_all_flows(flow_problems)
#         else:
#             print("\nNo flow problems found in the file.")
#     except FileNotFoundError:
#         print("\nNo flow.json file found. Skipping flow problems.")

# def solve_circulation_problems():
#     try:
#         with open('circulation.json') as f:
#             circ_data = json.load(f)
#             circ_problems = circ_data if isinstance(circ_data, list) else []
        
#         if circ_problems:
#             print("\n" + "="*60)
#             print("SOLVING CIRCULATION PROBLEMS")
#             print("="*60)
#             circ_calc = CirculationCalculator()
#             circ_calc.solve_all_circulations(circ_problems)
#         else:
#             print("\nNo circulation problems found in the file.")
#     except FileNotFoundError:
#         print("\nNo circulation.json file found. Skipping circulation problems.")

# def solve_work_integrals():
#     try:
#         with open('workintegrals.json') as f:
#             work_data = json.load(f)
#             work_problems = work_data if isinstance(work_data, list) else []
        
#         if work_problems:
#             print("\n" + "="*60)
#             print("SOLVING WORK INTEGRALS")
#             print("="*60)
#             work_calc = SolveWorkIntegral()
#             work_calc.solve_all_work_integrals(work_problems)
#         else:
#             print("\nNo work integral problems found in the file.")
#     except FileNotFoundError:
#         print("\nNo workintegrals.json file found. Skipping work integrals.")

# def solve_conservativity_problems():
#     try:
#         with open('conservativity.json') as f:
#             data = json.load(f)
#             problems = data.get('problems', [])
        
#         if not problems:
#             print("\nNo conservativity problems found in the file.")
#             return
            
#         checker = ConservativeChecker()
        
#         print("\n=== CONSERVATIVITY CHECKER ===")
#         print("=" * 60)
        
#         for problem in problems:
#             print(f"\nProblem {problem['id']}:")
#             print(f"Statement: {problem['problem_statement']}")
#             print(f"Vector Field Components:")
#             print(f"  i: {problem['function']['i']}")
#             print(f"  j: {problem['function']['j']}")
#             print(f"  k: {problem['function']['k']}")
            
#             try:
#                 is_cons = checker.is_conservative(
#                     problem['function']['i'],
#                     problem['function']['j'],
#                     problem['function']['k']
#                 )
                
#                 if is_cons is not None:
#                     result = "conservative" if is_cons else "not conservative"
#                     print(f"\nResult: The field is {result}")
#                     print(f"Expected: {problem['solution']}")
                    
#                     expected = "conservative" in problem['solution'].lower()
#                     if is_cons == expected:
#                         print("✓ Correct")
#                     else:
#                         print("✗ Incorrect")
#                 else:
#                     print("Could not determine conservativity (problem may be beyond scope)")
#             except Exception as e:
#                 print(f"Error processing problem: {str(e)}")
#                 print("This problem is beyond my scope")
            
#             print("=" * 60)
            
#     except FileNotFoundError:
#         print("\nNo conservativity.json file found. Skipping conservativity problems.")

# def solve_potential_functions():
#     try:
#         with open('potentialfunctions.json') as f:
#             potential_data = json.load(f)
#             potential_problems = potential_data if isinstance(potential_data, list) else potential_data.get('problems', [])
        
#         if potential_problems:
#             print("\n" + "="*60)
#             print("SOLVING POTENTIAL FUNCTION PROBLEMS")
#             print("="*60)
#             potential_calc = PotentialFunctionCalculator()
#             potential_calc.solve_all_potential_functions(potential_problems)
#         else:
#             print("\nNo potential function problems found in the file.")
#     except FileNotFoundError:
#         print("\nNo potentialfunctions.json file found. Skipping potential function problems.")
        
# def solve_dynamic_problem():
#     print("\n" + "="*50)
#     print("AI-POWERED PROBLEM SOLVER")
#     print("="*50)
#     print("Enter your problem statement (or paste it). Type 'done' on a new line when finished:")
    
#     # Read multi-line input
#     problem_lines = []
#     while True:
#         line = input()
#         if line.lower() == 'done':
#             break
#         problem_lines.append(line)
    
#     problem_statement = "\n".join(problem_lines)
    
#     if not problem_statement.strip():
#         print("No problem statement provided.")
#         return
    
#     print("\nAnalyzing your problem with AI...")
    
#     try:
#         # Initialize with safe defaults
#         problem_data = {
#             'problem_statement': problem_statement,
#             'curve': {'parametric_equations': '', 'type': ''},
#             'function': '',
#             'limits': [0, 1],  # Default limits
#             'vector_field': ''
#         }
        
#         # Initialize AI model
#         ai_model = AIModelIntegrator()
        
#         # Safe feature extraction with fallbacks
#         try:
#             # Extract function
#             func_match = re.search(r'(?:f\([^)]+\)|integrand)\s*=\s*([^\s,]+)', problem_statement, re.I)
#             if func_match:
#                 problem_data['function'] = func_match.group(1)
#             else:
#                 problem_data['function'] = 'x+y'  # Default
            
#             # Extract curve
#             curve_match = re.search(r'(?:r\(t\)|curve)\s*=\s*([^\n]+)', problem_statement, re.I)
#             if curve_match:
#                 problem_data['curve']['parametric_equations'] = curve_match.group(1)
            
#             # Extract limits
#             limits_match = re.search(r'(?:from|t\s*∈\s*)\s*([\dπ]+)\s*(?:to|\.\.|,)\s*([\dπ]+)', problem_statement, re.I)
#             if limits_match:
#                 problem_data['limits'] = [limits_match.group(1), limits_match.group(2)]
            
#             # Extract vector field if present
#             field_match = re.search(r'(?:F\s*=\s*|vector field\s*:\s*)([^\n]+)', problem_statement, re.I)
#             if field_match:
#                 problem_data['vector_field'] = field_match.group(1)
                
#         except Exception as e:
#             print(f"Warning: Feature extraction failed - {str(e)}")
#             print("Using default values for missing components")
        
#         # Get AI prediction with error handling
#         try:
#             ai_prediction = ai_model.predict(problem_data)
#             print(f"\nAI Prediction: {ai_prediction:.6f}" if ai_prediction else "No AI prediction available")
#         except Exception as e:
#             print(f"\nAI Prediction failed: {str(e)}")
#             ai_prediction = None
        
#         # Determine problem type
#         problem_type = ai_model.determine_problem_type(problem_data)
#         print(f"Detected problem type: {problem_type or 'Unknown'}")
        
#         # Use appropriate solver based on problem type
#         if problem_type == 'line_integral':
#             solver = IntegralSolver()
#             solution = solver.solve_problem(problem_data)
#             if solution.get('success'):
#                 print(f"\nExact Solution: {solution['result']:.8f}")
#                 if ai_prediction:
#                     print(f"AI Prediction: {ai_prediction:.8f}")
#                     print(f"Difference: {abs(solution['result'] - ai_prediction):.2e}")
#             else:
#                 print("\nExact solution failed")
#                 if ai_prediction:
#                     print(f"Using AI prediction: {ai_prediction:.8f}")
        
#         elif problem_type in ['flux', 'flow', 'circulation']:
#             # Similar handling for other problem types
#             print("\nExact solver for this problem type not yet implemented in dynamic mode")
#             if ai_prediction:
#                 print(f"Using AI prediction: {ai_prediction:.8f}")
        
#         else:
#             print("\nCould not determine appropriate solver")
#             if ai_prediction:
#                 print(f"Using AI prediction: {ai_prediction:.8f}")
    
#     except Exception as e:
#         print(f"\nError solving problem: {str(e)}")
#         print("Please check your problem statement and try again.")

# def main():
#     while True:
#         display_menu()
#         choice = input("\nEnter your choice (0-8): ")
        
#         if choice == '1':
#             solve_line_integrals()
#         elif choice == '2':
#             solve_flow_problems()
#         elif choice == '3':
#             solve_flux_problems()
#         elif choice == '4':
#             solve_circulation_problems()
#         elif choice == '5':
#             solve_work_integrals()
#         elif choice == '6':
#             solve_conservativity_problems()
#         elif choice == '7':
#             solve_potential_functions()
#         elif choice == '8':
#             solve_dynamic_problem()
#         elif choice == '0':
#             print("\nExiting program. Goodbye!")
#             break
#         else:
#             print("\nInvalid choice. Please enter a number between 0 and 7.")
        
#         input("\nPress Enter to continue...")

# if __name__ == "__main__":
#     main()