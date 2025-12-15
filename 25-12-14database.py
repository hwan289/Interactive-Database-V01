import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import io
import contextlib
import streamlit.components.v1 as components

# Advanced Stats Imports
try:
    import sklearn
    from sklearn import preprocessing,decomposition, cluster, linear_model, ensemble
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

try:
    import puter
    HAS_PUTER_SDK = True
except ImportError:
    HAS_PUTER_SDK = False

HAS_R = False # R support removed for Pure Python deployment

# --- Page Config ---
st.set_page_config(page_title="Generative Analytics", page_icon="📊", layout="wide")

# --- CSS ---
st.markdown("""
<style>
    .stTextArea textarea { font-family: 'Consolas', monospace; font-size: 13px; background-color: #f4f6f9; }
    .stAlert { padding: 0.5rem; }
    .chat-row { display: flex; flex-direction: column; margin-bottom: 20px; }
    .chat-bubble { padding: 15px; border-radius: 10px; max-width: 100%; word-wrap: break-word; }
    .user-msg { background-color: #f0f2f6; align-self: flex-end; border-bottom-right-radius: 2px; }
    .bot-msg { background-color: #ffffff; border: 1px solid #e0e0e0; align-self: flex-start; border-bottom-left-radius: 2px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
    pre { background: #f8f9fa; color: #333333; padding: 10px; border-radius: 5px; overflow-x: auto; border: 1px solid #ddd; }
    
    /* Custom User Chat Design (Global) */
    .user-chat-container { display: flex; justify-content: flex-end; align-items: flex-start; margin: 10px 0; }
    .user-chat-bubble { background: #f0f2f6; color: #333; padding: 12px 16px; border-radius: 18px 18px 2px 18px; max-width: 75%; margin-right: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.1); font-size: 15px; }
    .user-avatar { width: 32px; height: 32px; border-radius: 50%; background: #ff4b4b; color: white; display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0; }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if 'history' not in st.session_state:
    st.session_state.history = []  # List of dicts: {'role': 'user'/'assistant', 'content': str, 'code': str, 'result': any}
if 'input_buffer' not in st.session_state:
    st.session_state.input_buffer = ""

# --- Helper Functions ---

def execute_python_code(code, df):
    """Executes Python code in a safe local environment and captures output + figures."""
    import io
    import contextlib
    
    # Capture Output
    output_buffer = io.StringIO()
    figs = []
    
    # Common Env Setup
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Custom context for plots
    def show_plot(fig=None):
        if fig: figs.append(fig)
        else: figs.append(plt.gcf())
        plt.clf() # Clear current figure
        
    local_env = {
        'df': df, 
        'data': df, # Alias for robustness
        'pd': pd, 
        'np': np, 
        'plt': plt, 
        'sns': sns, 
        'st': st,
        'show': show_plot 
    }
    
    if HAS_SKLEARN:
        local_env['sklearn'] = sklearn
    if HAS_STATSMODELS:
        local_env['sm'] = sm
        local_env['smf'] = smf
    if HAS_R:
        local_env['robjects'] = robjects
        local_env['pandas2ri'] = pandas2ri
        local_env['localconverter'] = localconverter


    
    try:
        # Clear any existing plots to avoid capturing old ones
        plt.clf()
        plt.close('all')
        
        # Set moderate figure size (approx 1/2 screen width)
        plt.rcParams['figure.figsize'] = (6, 4)
        plt.rcParams['figure.dpi'] = 100
        
        with contextlib.redirect_stdout(output_buffer):
            # Execution for Python
            # Wrap in try-except block for execution
            exec(code, local_env)
            
        # Check for open figures if show() wasn't called explicitly but plt was used
        # Check for open figures if show() wasn't called explicitly but plt was used
        if plt.get_fignums():
            # Save ALL open figures
            for i in plt.get_fignums():
                figs.append(plt.figure(i))
            plt.close('all') # Clear after capturing
        
        # Check for Result Table (DataFrame)
        if 'result_table' in local_env:
            # We treat tables as artifacts too
            figs.append({'type': 'table', 'data': local_env['result_table']})
            
        return True, output_buffer.getvalue(), figs, None
        
    except Exception as e:
        return False, output_buffer.getvalue(), figs, str(e)

def ai_generate_code(prompt, df, model, api_key, context_history=[]):
    """Generates code using Python SDK (Server-side Auto-Run)."""
    if not HAS_PUTER_SDK:
        return None, "Puter SDK not installed."
        
    df_head = df.head(3).to_markdown()
    columns = list(df.columns)
    
    system_prompt = f"""You are an expert data scientist.
    DataFrame 'df' Columns: {columns}
    Data Sample:
    {df_head}
    
    Tasks:
    1. Write Python code to solve the user's request.
    2. Use 'print()' to show text results.
    3. Create plots using 'plt' (matplotlib) or 'sns' (seaborn).
    4. **CRITICAL**: Do NOT call 'plt.show()' and do NOT call 'st.pyplot()'. Just create the plot and leave it open. I will capture and display it automatically.
    5. NO Markdown backticks. Just pure code.
    6. Be concise.
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add brief context
    for msg in context_history[-3:]: # Last 3 messages for context
        if msg['role'] == 'user':
            messages.append({"role": "user", "content": msg['content']})
        elif msg['role'] == 'assistant':
             messages.append({"role": "assistant", "content": msg.get('code', '') or msg['content']})
             
    messages.append({"role": "user", "content": prompt})

    try:
        resp = puter.ChatCompletion.create(
            model=model,
            messages=messages,
            api_key=api_key
        )
        # Parse Response
        content = ""
        if isinstance(resp, dict):
            content = resp.get('choices', [{}])[0].get('message', {}).get('content', '') or str(resp)
        else:
            content = str(resp)
            
        # Clean Code
        code = content.replace("```python", "").replace("```", "").strip()
        return code, None
    except Exception as e:
        return None, str(e)

def render_js_widget(df, language, initial_prompt=""):
    """Renders the Client-side JS Widget for 'No Key' mode."""
    columns = list(df.columns)
    data_head = df.head(3).to_markdown().replace("\n", "\\n").replace("`", "'")
    
    
    # Dynamic System Prompt based on Language
    if language == "Python":
        sys_libs = "Libraries: pandas (pd), numpy (np), matplotlib (plt), seaborn (sns), sklearn, statsmodels (sm, smf), tabulate."
        sys_extra = """Capabilities: Regression, Path Analysis, PCA, Factor Analysis, Clustering allowed.
                IMPORTANT: For Factor Analysis, use 'sklearn.decomposition.FactorAnalysis'. Do NOT use 'factor_analyzer'.
                NOTE: 'FactorAnalysis' object in sklearn might not have 'n_components_'. Use 'components_.shape[0]' instead.
                WARNING: sklearn FactorAnalysis does NOT have 'explained_variance_' or 'explained_variance_ratio_'. Do not access them.
                Style: Use 'sns.set_style("whitegrid")' (Seaborn) for all plots. 
                  - CRITICAL: Do NOT use 'plt.style.use("whitegrid")' (this crashes). Use 'sns.set_style'.
                Constraint: Use SMALL figure size (6, 4). Do NOT call plt.show(). just create figure.
                 Tables: 
                  - CRITICAL: Assign the final DataFrame to variable 'result_table'. This renders an INTERACTIVE EXCEL-LIKE table.
                  - Do NOT use 'to_html' or print HTML strings. 'result_table' MUST be a Pandas DataFrame object.
                  - Regression: Extract coefficients: 'result_table = pd.read_html(model.summary().tables[1].as_html(), flavor="html5lib", header=0, index_col=0)[0]'
                STRATEGY: PURE PYTHON ONLY.
                  - Use 'statsmodels' (smf.ols, smf.logit) for regression.
                  - Use 'sklearn' for ML/PCA/Clustering.
                  - CRITICAL: Do NOT import rpy2. Do NOT use R. PURE PYTHON ONLY."""
        code_example = """# [Python Code Block]
                import pandas as pd
                # ... code ..."""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://js.puter.com/v2/"></script>
        <script>
            // Global Error Handler (Must be first to catch parsing errors)
            window.onerror = function(message, source, lineno, colno, error) {{
                // Use a fallback div if result-area isn't ready
                let area = document.getElementById('result-area');
                if(!area) {{
                     area = document.createElement('div');
                     document.body.appendChild(area);
                }}
                area.innerHTML += `<div style="color:red; background:#fee; padding:5px; margin:5px 0; border:1px solid red; font-weight:bold;">JS Error: ${{message}} (Line ${{lineno}})</div>`;
            }};
        </script>
        <style>
            body {{ font-family: sans-serif; margin: 0; padding: 10px; background: #fff; border: 1px solid #eee; border-radius: 8px; box-sizing: border-box; }}
            .info {{ font-size: 12px; color: #666; margin-bottom: 8px; }}
            textarea {{ width: 100%; box-sizing: border-box; border: 1px solid #ccc; border-radius: 4px; padding: 8px; font-family: inherit; resize: vertical; margin-bottom: 8px; }}
            button {{ 
                background: #ff4b4b; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-weight: 600; 
                user-select: none; -webkit-user-select: none; /* Fix: Prevent text selection/Look Up menu */
            }}
            button:hover {{ background: #ff2b2b; }}
            button:active {{ transform: translateY(1px); }}
            pre {{ background: #f5f5f5; color: #333; padding: 10px; border-radius: 4px; overflow: auto; position: relative; margin-top: 5px; border: 1px solid #ddd; white-space: pre-wrap; word-wrap: break-word; }}
            .copy-btn {{ position: absolute; top: 5px; right: 5px; background: #ddd; color: #333; font-size: 10px; padding: 2px 6px; border-radius: 3px; border: none; cursor: pointer; }}
            .copy-btn:hover {{ background: #ccc; }}
            .explanation {{ background-color: #eef4ff; padding: 10px; border-radius: 5px; border-left: 4px solid #007bff; font-size: 13px; color: #2c3e50; margin-bottom: 10px; white-space: pre-wrap; line-height: 1.6; word-wrap: break-word; }}
            
            /* Chips / Suggestions */
            .chips-container {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }}
            .chip {{ 
                background: #f0f2f6; border: 1px solid #e0e0e0; border-radius: 16px; 
                padding: 4px 12px; font-size: 11px; color: #555; cursor: pointer; transition: all 0.2s;
            }}
            .chip:hover {{ background: #eef4ff; border-color: #007bff; color: #007bff; }}
        </style>
    </head>
    <body>
        <div class="info"><b>1. Client-side Generation</b></div>
        
        <!-- Suggestions Area -->
        <div id="chips-area" class="chips-container"></div>

        <textarea id="prompt" rows="2" placeholder="Describe your request (e.g., 'Show me a regression table')...">{initial_prompt}</textarea>
        
        <div style="display:flex; gap:10px; align-items:center;">
             <button id="gen-btn" type="button" onclick="ask()">Generate Code</button>
             <span id="status" style="font-size:12px; color:green; display:none;">Generating...</span>
             <select id="model" style="margin-left:auto; padding:5px; max_width:180px;">
                <option value="gemini-2.5-flash-lite">Gemini 2.5 Flash Lite</option>
                <option value="gemini-2.5-flash">Gemini 2.5 Flash</option>
                <option value="gemini-2.5-pro">Gemini 2.5 Pro</option>
                <option value="gemini-1.5-pro">Gemini 1.5 Pro</option>
                <option value="gpt-4o-mini">GPT-4o Mini</option>
                <option value="gpt-4o">GPT-4o</option>
                <option value="claude-3-5-sonnet">Claude 3.5 Sonnet</option>
             </select>
        </div>
        
        <div id="result-area"></div>

        <script>
            console.log("Generative Widget Loaded");

            // Event Listeners removed - using onclick logic for safety
            
            // Initial Suggestions
            const starters = ["Show summary statistics", "Plot a correlation heatmap", "Analyze trends over time"];
            renderChips(starters);

            function renderChips(list) {{
                const con = document.getElementById('chips-area');
                if(!con) return;
                con.innerHTML = "";
                list.forEach(txt => {{
                    const el = document.createElement('div');
                    el.className = 'chip';
                    el.innerText = txt;
                    el.onclick = () => {{
                        document.getElementById('prompt').value = txt;
                        ask(); // Auto-submit
                    }};
                    con.appendChild(el);
                }});
            }}

            async function ask() {{
                console.log("Ask triggered");
                const p = document.getElementById('prompt').value;
                const status = document.getElementById('status');
                const area = document.getElementById('result-area');
                
                if(!p) return;
                status.style.display = 'block';
                area.innerHTML = '';
                
                // DATA_HEAD injected from Python (sanitized)
                const sys = `Data Cols: {columns}. Sample: {data_head}. 
                Task: Write {language} Code. NO Explanation. Just Code.
                Dataset: The dataframe is loaded as 'df'. Use 'df'.
                {sys_libs}
                {sys_extra}
                
                IMPORTANT: Structure your response into two distinct parts:
                1. ANALYSIS: A brief explanation of what you are doing.
                2. CODE: The {language} code block.
                2. CODE: The {language} code block.
                3. SUGGESTIONS: A JSON list of 3 short follow-up questions. Format: ["Q1", "Q2", "Q3"].
                   IMPORTANT: Do NOT use numbered lists (1. 2. 3.). Use a pure JSON array.
                
                Example Format:
                ANALYSIS:
                Here is the correlation matrix...
                
                CODE:
                {code_example}
                
                SUGGESTIONS: ["Q1", "Q2", "Q3"]`;
                
                try {{
                    const model = document.getElementById('model').value;
                    const resp = await puter.ai.chat(sys + "\\nUser: " + p, {{ model: model }});
                    
                    let raw = null;
                    if (typeof resp === 'string') raw = resp;
                    else if (resp && typeof resp === 'object') {{
                        if (resp.message && resp.message.content) raw = resp.message.content;
                        else if (resp.choices && resp.choices[0] && resp.choices[0].message && resp.choices[0].message.content) raw = resp.choices[0].message.content;
                        else if (resp.text) raw = resp.text;
                    }}

                    if (!raw) {{
                        raw = JSON.stringify(resp);
                        if (resp && resp.usage && resp.usage.completion_tokens === 0) throw new Error("AI returned 0 tokens.");
                    }}
                    
                    // 1. Extract Suggestions (Robust RegExp)
                    // 1. Extract Suggestions (Simple String Parsing)
                    let suggestions = [];
                    // Look for SUGGESTIONS:
                    let matchIdx = raw.lastIndexOf("SUGGESTIONS:");
                    if (matchIdx > -1) {{
                        let jsonStr = raw.substring(matchIdx + 12).trim();
                        // Always remove the SUGGESTIONS block from the raw text so it defaults to hidden
                        // We do this BEFORE parsing so even if parse fails, the text is gone from UI
                        raw = raw.substring(0, matchIdx).trim();
                        
                        // Try to find [ ... ]
                        let openB = jsonStr.indexOf("[");
                        let closeB = jsonStr.lastIndexOf("]");
                        if (openB > -1 && closeB > openB) {{
                            try {{
                                suggestions = JSON.parse(jsonStr.substring(openB, closeB + 1));
                                renderChips(suggestions);
                            }} catch(e) {{ console.log("JSON Parse fail", e); }}
                        }}
                    }}

                    // 2. Strict Separation of Code and Analysis (String Method Version - Robust)
                    let code = "";
                    let explanation = "";
                    
                    // Strategy A: Explicit "CODE:" separator
                    if (raw.includes("CODE:")) {{
                        const portions = raw.split("CODE:");
                        explanation = portions[0];
                        // Everything after first CODE: is code
                        code = portions.slice(1).join("CODE:"); 
                    }} 
                    // Strategy B: Markdown Code Blocks
                    else if (raw.includes("```")) {{
                        // Find first code block
                        const firstTick = raw.indexOf("```");
                        // Check if it's python/r
                        const nextNewline = raw.indexOf("\\n", firstTick);
                        const endTick = raw.indexOf("```", nextNewline + 1);
                        
                        if (endTick > firstTick) {{
                            code = raw.substring(firstTick, endTick + 3); // Include ticks for now
                            // Explanation is everything BEFORE the code + everything AFTER
                            explanation = raw.replace(code, "");
                        }} else {{
                             // Unclosed block? Treat as all code if imported
                             if (raw.includes("import ") || raw.includes("pd.")) code = raw;
                             else explanation = raw;
                        }}
                    }} 
                    else {{
                        // Strategy C: Heuristic
                        if ((raw.includes("import ") || raw.includes("pd.")) && !raw.includes("ANALYSIS:")) {{
                             code = raw;
                        }} else {{
                             explanation = raw;
                        }}
                    }}
                    
                    // Cleanup
                    // Remove "ANALYSIS:" label
                    explanation = explanation.replace("ANALYSIS:", "").trim();
                    
                    // Remove markdown ticks from code using Regex
                    code = code.trim();
                    // Regex to strip start ```python... and end ```
                    code = code.replace(/^```[a-z]*\s*/i, "").replace(/```\s*$/i, "").trim();
                    code = code.trim();

                    if(code) code = `# Run this in {language}\\n` + code;
                    
                    let html = "";
                    
                    // Render Analysis (Text)
                    if (explanation) {{
                        html += `<div class="explanation"><b>💡 Analysis:</b><br>${{explanation}}</div>`;
                    }}
                    
                    // Render Code (Box)
                    if (code) {{
                        // Apply Simple Syntax Highlighting (Robust RegExp)
                        // Note: formatting \\\\b ensures it becomes \\b in JS string, which is \b in RegExp
                        let coloredCode = code.replace(/</g, "&lt;").replace(/>/g, "&gt;");
                        
                        try {{
                            coloredCode = coloredCode
                                .replace(new RegExp("^(#.*)$", "gm"), "<span class='comment'>$1</span>")
                                .replace(new RegExp("\\\\b(import|from|def|return|if|else|for|while|try|except|with|as)\\\\b", "g"), "<span class='keyword'>$1</span>")
                                .replace(new RegExp("\\\\b(plt\\\\.|sns\\\\.|pd\\\\.|np\\\\.|st\\\\.)(\\\\w+)", "g"), "$1<span class='func'>$2</span>");
                        }} catch(e) {{ console.log("Highlight error", e); }}

                        // Safe HTML Generation (Avoid backticks in JS source)
                        html += '<div style="font-size:12px; color:#666; margin-top:10px; margin-bottom:5px;"><b>2. Copy & Paste 👇:</b></div>';
                        html += '<pre data-code="' + encodeURIComponent(code) + '">' + coloredCode + '<button class="copy-btn" onclick="copy(this)">COPY</button></pre>';
                    }}
                    
                    if (!html) html = `<div style="color:orange">AI returned empty content.</div>`;
                    area.innerHTML = html;

                }} catch(e) {{
                    area.innerHTML = '<div style="color:red; margin-top:10px;">Error: ' + e + '</div>';
                }} finally {{
                    status.style.display = 'none';
                }}
            }}
            
            function copy(btn) {{
                const pre = btn.parentNode;
                const rawCode = decodeURIComponent(pre.getAttribute('data-code'));
                navigator.clipboard.writeText(rawCode);
                btn.innerText = "COPIED";
            }}
            
            // We moved Login to sidebar, so widget is pure gen now
        </script>
    <div style="font-size:12px; margin-top:5px; color:#888;">
        <a href="https://puter.com" target="_blank" style="color:#666; text-decoration:none;">Check Quota / Usage</a>
    </div>
    </body>
    </html>
    """
    components.html(html, height=450, scrolling=True)

def render_sidebar_auth():
    """Renders a small Auth component in the sidebar."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://js.puter.com/v2/"></script>
        <style>
            body { font-family: sans-serif; margin: 0; padding: 0; }
            .btn { width: 100%; background: #007bff; color: white; border: none; padding: 5px; border-radius: 4px; cursor: pointer; font-size:12px; }
            .btn-out { background: #6c757d; }
            .user { font-size: 12px; margin-bottom: 5px; font-weight: bold; color: #333; }
        </style>
    </head>
    <body>
        <div id="auth-area">checking...</div>
        <script>
            async function init() {
                const user = await puter.auth.getUser();
                const div = document.getElementById('auth-area');
                if(user) {
                    div.innerHTML = `<div class='user'>👤 ${user.username}</div><button class='btn' onclick="window.open('https://puter.com', '_blank')" style="margin-bottom:5px; background:#17a2b8;">Manage Account</button><button class='btn btn-out' onclick='logout()'>Log Out</button>`;
                } else {
                    div.innerHTML = `<button class='btn' onclick='login()'>Sign In to Puter</button>`;
                }
            }
            async function login() { await puter.auth.signIn(); init(); }
            async function logout() { await puter.auth.signOut(); init(); }
            init();
        </script>
    </body>
    </html>
    """
    components.html(html, height=120)

# --- Sidebar & Setup ---

with st.sidebar:
    st.title("Settings")
    

    
    st.divider()
    
    # 1. Execution Engine (Restored)
    # language = st.radio("Execution Engine", ["Python", "R"])
    language = "Python" # R option hidden per user request
    st.markdown(f"**Execution Engine:** {language}")
    
    st.divider()
    
    # 2. Dataset
    st.subheader("Dataset")
    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded:
        st.session_state.df = pd.read_csv(uploaded)
    else:
        # Default Data
        if 'df' not in st.session_state:
            # Generate 5 Related Tables (Retail Analytics Scenario)
            np.random.seed(42)
            n_rows = 500
            
            # 1. Customers
            customer_ids = [f"C{i:03d}" for i in range(100)]
            customers = pd.DataFrame({
                'CustomerID': customer_ids,
                'Age': np.random.randint(18, 70, 100),
                'Gender': np.random.choice(['M', 'F'], 100),
                'Location': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], 100),
                'Membership_Level': np.random.choice(['Bronze', 'Silver', 'Gold'], 100, p=[0.5, 0.3, 0.2])
            })

            # 2. Products
            product_ids = [f"P{i:02d}" for i in range(20)]
            products = pd.DataFrame({
                'ProductID': product_ids,
                'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], 20),
                'Cost': np.random.uniform(10, 100, 20),
                'Price': np.random.uniform(20, 200, 20)
            })
            
            # 3. Marketing Campaigns (Influence Sales)
            campaigns = ['Summer_Sale', 'Black_Friday', 'Holiday_Special', 'None']
            dates = pd.date_range(start="2023-01-01", periods=n_rows)
            
            # 4. Orders (Linking Customers, Dates, Campaigns)
            orders = pd.DataFrame({
                'OrderID': [f"O{i:04d}" for i in range(n_rows)],
                'Date': dates,
                'CustomerID': np.random.choice(customer_ids, n_rows),
                'Campaign': np.random.choice(campaigns, n_rows, p=[0.2, 0.2, 0.2, 0.4])
            })
            
            # 5. Order Lines (Linking Orders and Products with Quantity)
            # To simulate insight: Gold members buy more, Campaigns boost quantity
            order_lines = []
            for i in range(n_rows):
                p_id = np.random.choice(product_ids)
                qty = np.random.randint(1, 5)
                
                # Logic: Boost quantity if campaign is active
                if orders.iloc[i]['Campaign'] != 'None':
                    qty += np.random.randint(0, 3)
                    
                order_lines.append({
                    'OrderID': orders.iloc[i]['OrderID'],
                    'ProductID': p_id,
                    'Quantity': qty
                })
            order_lines_df = pd.DataFrame(order_lines)

            # Merge into one Master Table for Analysis
            df_merged = orders.merge(order_lines_df, on='OrderID')
            df_merged = df_merged.merge(customers, on='CustomerID')
            df_merged = df_merged.merge(products, on='ProductID')
            
            # Derived Metrics
            df_merged['Revenue'] = df_merged['Price'] * df_merged['Quantity']
            df_merged['Profit'] = (df_merged['Price'] - df_merged['Cost']) * df_merged['Quantity']
            
            st.session_state.df = df_merged

    # Sidebar Auth
    st.divider()
    st.caption("Puter Account")
    render_sidebar_auth()
    
    st.dataframe(st.session_state.df.head(5), use_container_width=True)
    
    if st.button("Clear Chat History"):
        st.session_state.history = []
        st.rerun()

# --- Main Interaction ---

st.title("Generative Analytics")

# 1. Display History (Julius Style)
for i, msg in enumerate(st.session_state.history):
    role = msg['role']
    content = msg.get('content', '')
    code = msg.get('code')
    result = msg.get('result') # (success, output, figs, error)
    
    # Custom Rendering to put User Avatar on the RIGHT
    if role == 'user':
        st.markdown(f"""
        <div class="user-chat-container">
            <div class="user-chat-bubble">{content}</div>
            <div class="user-avatar">
                <svg viewBox="0 0 24 24" width="20" height="20" fill="white">
                    <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                </svg>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    else: # Assistant
        with st.chat_message(role):
            if content:
                st.markdown(content)
            
            if code:
                with st.expander("Generated Code", expanded=False):
                    st.code(code, language='python')
                
        if result:
            success, output, figs, error = result
            if output:
                st.text(output)
            if figs:
                for f in figs: 
                    # Check if it's a table
                    if isinstance(f, dict) and f.get('type') == 'table':
                        data = f['data']
                        # Handle Statsmodels Summary or arbitrary objects
                        if isinstance(data, (pd.DataFrame, pd.Series)):
                            st.dataframe(data)
                        elif hasattr(data, 'as_html'): # Statsmodels Summary
                            st.markdown(data.as_html(), unsafe_allow_html=True)
                        else:
                            st.text(str(data))
                    else:
                        # Use columns to restrict width to 1/3
                        c1, _ = st.columns([1, 2])
                        with c1:
                            st.pyplot(f, use_container_width=True)
            if not success and error:
                st.error(f"Execution Error: {error}")

# 2. Suggestions & Input Area

def get_suggestions(history):
    """Returns context-aware prompt suggestions based on history."""
    if not history:
        return ["Analyze basic statistics", "Show correlation matrix", "Visualize sales trends"]
    
    last_msg = history[-1]['content'].lower()
    
    if "regression" in last_msg or "model" in last_msg:
        return ["Check model residuals", "Show coefficients table", "Predict on new data"]
    elif "plot" in last_msg or "graph" in last_msg or "chart" in last_msg:
        return ["Change plot style", "Save this figure", "Analyze outliers"]
    elif "correlation" in last_msg:
        return ["Run Factor Analysis", "Filter for high correlation", "Visualize heatmap"]
    elif "factor" in last_msg or "pca" in last_msg:
        return ["Scree plot", "Show factor loadings", "Cluster the results"]
    
    return ["Summarize these results", "Create a dashboard", "Drill down by category"]

# Render Suggestions above input
suggs = get_suggestions(st.session_state.history)
cols = st.columns(len(suggs))
for i, sugg in enumerate(suggs):
    if cols[i].button(sugg, use_container_width=True, type="secondary"):
        st.session_state.history.append({"role": "user", "content": sugg})
        st.rerun()

prompt = st.chat_input("Ask about your data...")

# 3. Processing Logic
if prompt:
    # Append User Message
    st.session_state.history.append({"role": "user", "content": prompt})
    st.rerun()

# Handle pending action (last message is user)
if st.session_state.history and st.session_state.history[-1]['role'] == 'user':
    last_prompt = st.session_state.history[-1]['content']
    
    # --- MANUAL MODE (JS only now) ---
    with st.chat_message("assistant"):
        st.info(f"Use the widget below to generate {language} code, then **Paste it** into the box to run.")
        
        # 1. Render JS Widget for "Client-side generation"
        render_js_widget(st.session_state.df, language, initial_prompt=last_prompt)
        
        # 2. Form for execution
        with st.form("manual_exec_form"):
            # Add 'key' based on language to auto-clear buffer when switching engines
            pasted_code = st.text_area("3. Paste Code Here & Run", height=150, key=f"exec_code_{language}")
            submitted = st.form_submit_button("Run Code")
            
            if submitted and pasted_code:
                # Init defaults
                success, out, figs, exec_err = False, "", [], None
                
                # Unified Execution (since R mode produces Python code now)
                # We remove the if-check to ensure it runs regardless of language label quirks
                success, out, figs, exec_err = execute_python_code(pasted_code, st.session_state.df)
                
                # Debug feedback (remove later if annoyance)
                st.success("Execution Completed")
                
                if exec_err:
                     st.error(f"Execution Error: {exec_err}")
                
                # Need to escape the form to save history properly, but st.session_state works
                st.session_state.history.append({
                    "role": "assistant",
                    "content": f"Executed Manual {language} Code:",
                    "code": pasted_code,
                    "result": (success, out, figs, exec_err)
                })
                st.rerun()