<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sercomm Tool Suite</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/xlsx/0.18.5/xlsx.full.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .view { transition: opacity 0.3s ease-in-out; }
        .hidden { display: none; }
        #message-box {
            position: fixed; top: 20px; right: 20px; padding: 1rem; border-radius: 0.5rem; color: white;
            z-index: 1000; opacity: 0; transform: translateY(-20px); transition: opacity 0.3s, transform 0.3s;
        }
        #message-box.show { opacity: 1; transform: translateY(0); }
        #message-box.success { background-color: #28a745; }
        #message-box.error { background-color: #dc3545; }
        #message-box.info { background-color: #17a2b8; }
        .st-tabs-container { display: flex; border-bottom: 2px solid #4a5568; margin-bottom: 1.5rem; }
        .st-tab { padding: 0.75rem 1.5rem; cursor: pointer; color: #a0aec0; border-bottom: 2px solid transparent; transform: translateY(2px); transition: color 0.2s, border-color 0.2s; }
        .st-tab.active { color: #e2e8f0; border-color: #4299e1; }
        .st-tab-content { display: none; animation: fadeIn 0.5s; }
        .st-tab-content.active { display: block; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .data-editor-table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        .data-editor-table th, .data-editor-table td { border: 1px solid #4a5568; padding: 0.5rem; text-align: left; }
        .data-editor-table th { background-color: #2d3748; }
        .data-editor-table input, .data-editor-table select { background-color: #1a202c; border: 1px solid #4a5568; width: 100%; padding: 0.25rem; border-radius: 0.25rem;}
        input[type=number]::-webkit-inner-spin-button, 
        input[type=number]::-webkit-outer-spin-button { 
            -webkit-appearance: none; margin: 0; 
        }
        input[type=number] { -moz-appearance: textfield; }
    </style>
</head>
<body class="bg-slate-900 text-slate-200">

    <!-- Global Loader -->
    <div id="loader" class="fixed inset-0 bg-slate-900 flex flex-col items-center justify-center z-50">
        <svg class="animate-spin h-10 w-10 text-cyan-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
        <p class="mt-4 text-slate-400">Initializing System...</p>
    </div>

    <!-- Message Box -->
    <div id="message-box"></div>

    <!-- App Wrapper -->
    <div id="app-wrapper" class="w-full h-screen hidden">
        
        <!-- Auth View -->
        <div id="auth-view" class="view flex items-center justify-center h-full">
            <div class="w-full max-w-md mx-auto">
                <div class="bg-slate-800 p-8 rounded-xl shadow-2xl">
                    <div class="text-center mb-8">
                        <h1 class="text-3xl font-bold text-white">Sercomm Tool Suite</h1>
                        <p class="text-slate-400">Thermal Engineering Platform</p>
                    </div>
                    <form id="login-form">
                        <div class="mb-4">
                            <label for="login-email" class="block text-sm font-medium text-slate-300 mb-2">Company Email</label>
                            <input type="email" id="login-email" required class="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-500">
                        </div>
                        <div class="mb-6">
                            <label for="login-password" class="block text-sm font-medium text-slate-300 mb-2">Password</label>
                            <input type="password" id="login-password" required class="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-500">
                        </div>
                        <button type="submit" class="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-bold py-2 px-4 rounded-lg transition duration-300">Login</button>
                    </form>
                    <div class="text-center mt-6">
                        <p class="text-sm text-slate-400">New user? <a href="#" id="show-register" class="font-medium text-cyan-400 hover:text-cyan-300">Request an account</a></p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Register View -->
        <div id="register-view" class="view hidden flex items-center justify-center h-full">
            <div class="w-full max-w-md mx-auto">
                <div class="bg-slate-800 p-8 rounded-xl shadow-2xl">
                    <div class="text-center mb-8">
                        <h1 class="text-2xl font-bold text-white">Create New Account</h1>
                        <p class="text-slate-400">Please use your company email</p>
                    </div>
                    <form id="register-form">
                        <div class="mb-4">
                            <label for="register-email" class="block text-sm font-medium text-slate-300 mb-2">Company Email</label>
                            <input type="email" id="register-email" required class="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-500">
                        </div>
                        <div class="mb-4">
                            <label for="register-password" class="block text-sm font-medium text-slate-300 mb-2">Set Password</label>
                            <input type="password" id="register-password" required minlength="6" class="w-full px-4 py-2 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-500">
                        </div>
                        <button type="submit" class="w-full bg-cyan-600 hover:bg-cyan-700 text-white font-bold py-2 px-4 rounded-lg transition duration-300">Submit Application</button>
                    </form>
                    <div class="text-center mt-6">
                        <p class="text-sm text-slate-400">Already have an account? <a href="#" id="show-login" class="font-medium text-cyan-400 hover:text-cyan-300">Back to Login</a></p>
                    </div>
                </div>
            </div>
        </div>

        <!-- Main App View (Sercomm Suite) -->
        <div id="main-app-view" class="view hidden h-screen flex">
            <!-- Sidebar -->
            <aside class="w-64 bg-slate-800 p-6 flex flex-col">
                <h2 class="text-2xl font-bold text-white mb-8">Sercomm Thermal Engineering</h2>
                <nav class="flex flex-col space-y-2">
                    <a href="#" id="tool-cobra" class="tool-selector active flex items-center p-3 rounded-lg bg-slate-700 text-white">
                         <svg class="w-6 h-6 mr-3" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M40 95 C 20 95, 10 75, 25 60 C 35 50, 65 50, 75 60 C 90 75, 80 95, 60 95 L 40 95 Z" fill="#2C7873"/><path d="M50 70 C 25 70, 25 40, 50 20 C 75 40, 75 70, 50 70 Z" fill="#4A938E" stroke="#FFFFFF" stroke-width="2"/><path d="M50 20 C 40 30, 40 50, 50 60 C 60 50, 60 30, 50 20" fill="#1E1E1E"/><circle cx="46" cy="45" r="4" fill="red"/><circle cx="54" cy="45" r="4" fill="red"/><path d="M48 62 L52 62 L50 70 Z" fill="#FFD700"/></svg>
                        Cobra
                    </a>
                    <a href="#" id="tool-viper" class="tool-selector flex items-center p-3 rounded-lg hover:bg-slate-700/50">
                        <svg class="w-6 h-6 mr-3" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M50 10 L85 45 L50 90 L15 45 Z" fill="#1E1E1E" stroke="#FF5733" stroke-width="4"/><path d="M50 25 C 40 35, 40 55, 50 65" stroke="#FFC300" stroke-width="5" stroke-linecap="round" fill="none"/><path d="M50 25 C 60 35, 60 55, 50 65" stroke="#FFC300" stroke-width="5" stroke-linecap="round" fill="none"/><path d="M42 45 L58 45" stroke="#FFC300" stroke-width="5" stroke-linecap="round"/><circle cx="40" cy="35" r="4" fill="#FFFFFF"/><circle cx="60" cy="35" r="4" fill="#FFFFFF"/></svg>
                        Viper
                    </a>
                </nav>
                <div class="mt-auto">
                     <p class="text-sm text-slate-400" id="user-email-display"></p>
                     <button id="admin-panel-button" class="hidden w-full mt-2 bg-amber-500 hover:bg-amber-600 text-white font-bold py-2 px-4 rounded-lg text-sm transition">Admin Panel</button>
                     <button id="logout-button" class="w-full mt-2 bg-red-600 hover:bg-red-700 text-white font-bold py-2 px-4 rounded-lg text-sm transition">Logout</button>
                </div>
            </aside>

            <!-- Main Content Area -->
            <main class="flex-1 p-8 overflow-y-auto">
                <!-- Cobra UI -->
                <div id="cobra-ui" class="tool-ui"></div>
                <!-- Viper UI -->
                <div id="viper-ui" class="tool-ui hidden"></div>
            </main>
        </div>

        <!-- Admin Panel View -->
        <div id="admin-panel-view" class="view hidden p-8">
            <header class="bg-slate-800/50 backdrop-blur-sm rounded-xl p-4 mb-6 flex justify-between items-center shadow-lg">
                <h1 class="text-xl font-bold">Admin Panel</h1>
                <button id="back-to-app-button" class="bg-cyan-600 hover:bg-cyan-700 text-white font-bold py-2 px-4 rounded-lg text-sm transition">Back to Main System</button>
            </header>
            <main class="bg-slate-800 p-8 rounded-xl shadow-lg">
                <h2 class="text-2xl font-bold mb-6">User Account Management</h2>
                <div class="overflow-x-auto"><table class="min-w-full divide-y divide-slate-700"><thead class="bg-slate-700/50"><tr><th class="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">Email</th><th class="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">Role</th><th class="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">Status</th><th class="px-6 py-3 text-left text-xs font-medium text-slate-300 uppercase tracking-wider">Actions</th></tr></thead><tbody id="user-list-body" class="bg-slate-800 divide-y divide-slate-700"></tbody></table></div>
            </main>
        </div>
    </div>

    <!-- MAIN SCRIPT -->
    <script type="module">
        // --- Firebase Authentication Module ---
        import { initializeApp } from "https://www.gstatic.com/firebasejs/11.6.1/firebase-app.js";
        import { getAuth, onAuthStateChanged, createUserWithEmailAndPassword, signInWithEmailAndPassword, signOut, sendEmailVerification, signInWithCustomToken } from "https://www.gstatic.com/firebasejs/11.6.1/firebase-auth.js";
        import { getFirestore, doc, setDoc, getDoc, collection, onSnapshot, updateDoc, serverTimestamp, query } from "https://www.gstatic.com/firebasejs/11.6.1/firebase-firestore.js";

        let currentUserData = null;
        let auth, db, appId;
        const loader = document.getElementById('loader');
        const appWrapper = document.getElementById('app-wrapper');
        const views = { auth: document.getElementById('auth-view'), register: document.getElementById('register-view'), main: document.getElementById('main-app-view'), admin: document.getElementById('admin-panel-view') };
        const messageBox = document.getElementById('message-box');

        function showView(viewName) {
            Object.keys(views).forEach(key => views[key].classList.toggle('hidden', key !== viewName));
            appWrapper.classList.remove('hidden');
            loader.classList.add('hidden');
        }

        function showMessage(text, type = 'info') {
            messageBox.textContent = text;
            messageBox.className = 'show';
            messageBox.classList.add(type);
            setTimeout(() => messageBox.classList.remove('show'), 5000);
        }

        async function initializeAndRunApp() {
            const firebaseConfigStr = typeof __firebase_config__ !== 'undefined' ? __firebase_config__ : null;
            if (!firebaseConfigStr) { loader.innerHTML = '<p class="text-red-400">Error: App configuration is missing.</p>'; return; }
            const firebaseConfig = JSON.parse(firebaseConfigStr);
            if (!firebaseConfig.apiKey) { loader.innerHTML = '<p class="text-red-400">Error: App configuration is invalid.</p>'; return; }
            appId = typeof __app_id__ !== 'undefined' ? __app_id__ : 'sercomm-tool-suite';
            const app = initializeApp(firebaseConfig);
            auth = getAuth(app);
            db = getFirestore(app);
            try {
                const token = typeof __initial_auth_token__ !== 'undefined' ? __initial_auth_token__ : null;
                if (token) await signInWithCustomToken(auth, token);
            } catch (error) { console.error("Auto sign-in failed:", error); showMessage("Auto sign-in failed. Please log in manually.", "error"); }
            onAuthStateChanged(auth, user => {
                if (user) {
                    if (!user.emailVerified && user.email) { showMessage("Please verify your email address, then log in again.", "info"); signOut(auth); showView('auth'); return; }
                    checkUserStatusAndData(user);
                } else { currentUserData = null; showView('auth'); }
            });
            setupAuthEventListeners();
        }

        async function checkUserStatusAndData(user) {
            if (!user.email) { showView('auth'); return; }
            const userDocRef = doc(db, `/artifacts/${appId}/public/data/users`, user.uid);
            try {
                const docSnap = await getDoc(userDocRef);
                if (docSnap.exists() && docSnap.data().isActive) {
                    currentUserData = { uid: user.uid, ...docSnap.data() };
                    document.getElementById('user-email-display').textContent = currentUserData.email;
                    document.getElementById('admin-panel-button').classList.toggle('hidden', currentUserData.role !== 'admin');
                    if (currentUserData.role === 'admin') listenForUserUpdates();
                    showView('main');
                    initializeSercommSuite(currentUserData);
                } else { showMessage("Your account is pending admin approval.", "error"); signOut(auth); }
            } catch (error) { console.error("Error fetching user data:", error); showMessage("Error reading user data.", "error"); signOut(auth); }
            finally { loader.classList.add('hidden'); }
        }

        function setupAuthEventListeners() {
            document.getElementById('login-form').addEventListener('submit', async (e) => { e.preventDefault(); loader.classList.remove('hidden'); try { await signInWithEmailAndPassword(auth, document.getElementById('login-email').value, document.getElementById('login-password').value); } catch (error) { showMessage("Login failed. Please check credentials.", "error"); loader.classList.add('hidden'); } });
            document.getElementById('register-form').addEventListener('submit', async (e) => { e.preventDefault(); loader.classList.remove('hidden'); const email = document.getElementById('register-email').value, password = document.getElementById('register-password').value; try { const cred = await createUserWithEmailAndPassword(auth, email, password); await setDoc(doc(db, `/artifacts/${appId}/public/data/users`, cred.user.uid), { email: cred.user.email, role: 'user', isActive: false, createdAt: serverTimestamp() }); await sendEmailVerification(cred.user); showMessage("Application submitted! Please check your email to verify your address.", "success"); signOut(auth); showView('auth'); } catch (error) { showMessage(error.code === 'auth/email-already-in-use' ? "This email is already registered." : "Registration failed.", "error"); } finally { loader.classList.add('hidden'); } });
            document.getElementById('logout-button').addEventListener('click', () => signOut(auth));
            document.getElementById('show-register').addEventListener('click', (e) => { e.preventDefault(); showView('register'); });
            document.getElementById('show-login').addEventListener('click', (e) => { e.preventDefault(); showView('auth'); });
            document.getElementById('admin-panel-button').addEventListener('click', () => showView('admin'));
            document.getElementById('back-to-app-button').addEventListener('click', () => showView('main'));
        }

        let unsubscribeUserList; function listenForUserUpdates() { if (unsubscribeUserList) unsubscribeUserList(); const q = query(collection(db, `/artifacts/${appId}/public/data/users`)); unsubscribeUserList = onSnapshot(q, (snapshot) => { const userListBody = document.getElementById('user-list-body'); userListBody.innerHTML = ''; snapshot.forEach(docSnap => { const user = { id: docSnap.id, ...docSnap.data() }; const tr = document.createElement('tr'); const activeText = user.isActive ? 'Active' : 'Inactive'; const activeClass = user.isActive ? 'text-green-400' : 'text-yellow-400'; const toggleActiveText = user.isActive ? 'Deactivate' : 'Activate'; tr.innerHTML = `<td class="px-6 py-4 whitespace-nowrap text-sm text-slate-300">${user.email}</td><td class="px-6 py-4 whitespace-nowrap text-sm text-slate-300">${user.role}</td><td class="px-6 py-4 whitespace-nowrap text-sm font-medium ${activeClass}">${activeText}</td><td class="px-6 py-4 whitespace-nowrap text-sm font-medium space-x-2"><button data-id="${user.id}" data-active="${user.isActive}" class="toggle-active-btn text-white px-3 py-1 text-xs rounded transition ${user.isActive ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'}">${toggleActiveText}</button></td>`; userListBody.appendChild(tr); }); }); }
        document.getElementById('user-list-body').addEventListener('click', async (e) => { if (!e.target.classList.contains('toggle-active-btn')) return; const userId = e.target.dataset.id; if (userId === auth.currentUser.uid) { showMessage("You cannot change your own status.", "error"); return; } const newStatus = !(e.target.dataset.active === 'true'); await updateDoc(doc(db, `/artifacts/${appId}/public/data/users`, userId), { isActive: newStatus }); showMessage(`User status updated.`, "success"); });

        // --- Sercomm Tool Suite Application ---
        function initializeSercommSuite(user) {
            renderViperUI();
            renderCobraUI();
            document.querySelectorAll('.tool-selector').forEach(selector => {
                selector.addEventListener('click', (e) => {
                    e.preventDefault();
                    document.querySelectorAll('.tool-selector').forEach(s => s.classList.remove('active', 'bg-slate-700'));
                    selector.classList.add('active', 'bg-slate-700');
                    document.querySelectorAll('.tool-ui').forEach(ui => ui.classList.add('hidden'));
                    const toolId = selector.id.replace('tool-', '') + '-ui';
                    document.getElementById(toolId).classList.remove('hidden');
                });
            });
            document.getElementById('tool-viper').click();
        }
        
        // --- VIPER JAVASCRIPT ---
        function renderViperUI() {
             const viperUI = document.getElementById('viper-ui');
             const inputClasses = "w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-500";
             const labelClasses = "block text-sm font-medium text-slate-300 mb-1";
             viperUI.innerHTML = `<div class="flex items-center border-b-2 border-slate-700 pb-4 mb-6"><div class="mr-4"><svg width="50" height="50" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M50 10 L85 45 L50 90 L15 45 Z" fill="#1E1E1E" stroke="#FF5733" stroke-width="4"/><path d="M50 25 C 40 35, 40 55, 50 65" stroke="#FFC300" stroke-width="5" stroke-linecap="round" fill="none"/><path d="M50 25 C 60 35, 60 55, 50 65" stroke="#FFC300" stroke-width="5" stroke-linecap="round" fill="none"/><path d="M42 45 L58 45" stroke="#FFC300" stroke-width="5" stroke-linecap="round"/><circle cx="40" cy="35" r="4" fill="#FFFFFF"/><circle cx="60" cy="35" r="4" fill="#FFFFFF"/></svg></div><div><h1 class="text-3xl font-bold text-white mb-0">Viper</h1><p class="text-slate-400 mt-0">Risk Analysis</p></div></div><div id="viper-tabs" class="st-tabs-container"><button data-tab="nat-conv" class="st-tab active">🍃 Natural Convection</button><button data-tab="force-conv" class="st-tab">🌬️ Forced Convection</button><button data-tab="solar-rad" class="st-tab">☀️ Solar Radiation</button></div><div id="nat-conv-content" class="st-tab-content active"><h2 class="text-2xl font-bold mb-4">Passive Cooling Power Estimator</h2><div class="grid md:grid-cols-2 gap-8"><div><h3 class="text-xl font-semibold mb-3">Input Parameters</h3><div class="space-y-4"><div><label for="nc_mat" class="${labelClasses}">Enclosure Material</label><select id="nc_mat" class="${inputClasses}"><option value="Plastic (ABS/PC)">Plastic (ABS/PC)</option><option value="Aluminum (Anodized)">Aluminum (Anodized)</option></select></div><div><label class="${labelClasses}">Product Dimensions (mm)</label><div class="grid grid-cols-3 gap-2"><input type="number" id="nc_l" value="200.0" class="${inputClasses}" placeholder="Length (L)"><input type="number" id="nc_w" value="150.0" class="${inputClasses}" placeholder="Width (W)"><input type="number" id="nc_h" value="50.0" class="${inputClasses}" placeholder="Height (H)"></div></div><div><label class="${labelClasses}">Operating Conditions (°C)</label><div class="grid grid-cols-2 gap-2"><input type="number" id="nc_ta" value="25" class="${inputClasses}" placeholder="Ambient Temp (Ta)"><input type="number" id="nc_ts" value="50" class="${inputClasses}" placeholder="Max. Surface Temp (Ts)"></div></div></div></div><div class="bg-slate-800 p-6 rounded-lg"><h3 class="text-xl font-semibold mb-3">Evaluation Result</h3><div id="nc-results" class="space-y-4"></div></div></div></div><div id="force-conv-content" class="st-tab-content"><h2 class="text-2xl font-bold mb-4">Active Cooling Airflow Estimator</h2><div class="grid md:grid-cols-2 gap-8"><div><h3 class="text-xl font-semibold mb-3">Input Parameters</h3><div class="space-y-4"><div><label for="fc_power_q" class="${labelClasses}">Power to Dissipate (Q, W)</label><input type="number" id="fc_power_q" value="50.0" class="${inputClasses}"></div><div><label for="fc_tin" class="${labelClasses}">Inlet Air Temp (Tin, °C)</label><input type="number" id="fc_tin" value="25" class="${inputClasses}"></div><div><label for="fc_tout" class="${labelClasses}">Max. Outlet Temp (Tout, °C)</label><input type="number" id="fc_tout" value="45" class="${inputClasses}"></div></div></div><div class="bg-slate-800 p-6 rounded-lg"><h3 class="text-xl font-semibold mb-3">Evaluation Result</h3><div id="fc-results" class="space-y-4"></div></div></div></div><div id="solar-rad-content" class="st-tab-content"><h2 class="text-2xl font-bold mb-4">Solar Heat Gain Estimator</h2><div class="grid md:grid-cols-2 gap-8"><div><h3 class="text-xl font-semibold mb-3">Input Parameters</h3><div class="space-y-4"><div><label for="solar_mat" class="${labelClasses}">Enclosure Color/Finish</label><select id="solar_mat" class="${inputClasses}"><option value="0.25">White (Paint)</option><option value="0.40">Silver (Paint)</option><option value="0.80">Dark Gray</option><option value="0.95" selected>Black (Plastic/Paint)</option><option value="custom">Other...</option></select><input type="number" id="solar_alpha_custom" value="0.5" class="${inputClasses} mt-2 hidden" placeholder="Custom Absorptivity (α)"></div><div><label for="solar_area" class="${labelClasses}">Projected Surface Area (mm²)</label><input type="number" id="solar_area" value="30000" class="${inputClasses}"></div><div><label for="solar_irradiance" class="${labelClasses}">Solar Irradiance (W/m²)</label><input type="number" id="solar_irradiance" value="1000" class="${inputClasses}"></div></div></div><div class="bg-slate-800 p-6 rounded-lg"><h3 class="text-xl font-semibold mb-3">Evaluation Result</h3><div id="solar-results" class="space-y-4"></div></div></div></div>`;
            setupViperEventListeners();
        }
        
        function setupViperEventListeners() {
            const tabs = document.querySelectorAll('#viper-tabs .st-tab');
            const contents = document.querySelectorAll('#viper-ui .st-tab-content');
            tabs.forEach(tab => {
                tab.addEventListener('click', () => {
                    tabs.forEach(t => t.classList.remove('active'));
                    tab.classList.add('active');
                    const targetId = tab.dataset.tab + '-content';
                    contents.forEach(c => c.classList.toggle('active', c.id === targetId));
                });
            });
            ['nc_mat', 'nc_l', 'nc_w', 'nc_h', 'nc_ta', 'nc_ts'].forEach(id => document.getElementById(id)?.addEventListener('input', calculateAndDisplayNC));
            ['fc_power_q', 'fc_tin', 'fc_tout'].forEach(id => document.getElementById(id)?.addEventListener('input', calculateAndDisplayFC));
            ['solar_mat', 'solar_alpha_custom', 'solar_area', 'solar_irradiance'].forEach(id => document.getElementById(id)?.addEventListener('input', calculateAndDisplaySolar));
            document.getElementById('solar_mat').addEventListener('change', (e) => { document.getElementById('solar_alpha_custom').classList.toggle('hidden', e.target.value !== 'custom'); });
            calculateAndDisplayNC(); calculateAndDisplayFC(); calculateAndDisplaySolar();
        }
        
        function calculateAndDisplayNC() {
            const materials = { "Plastic (ABS/PC)": {emissivity: 0.90, k_uniform: 0.65}, "Aluminum (Anodized)": {emissivity: 0.85, k_uniform: 0.90} };
            const [L, W, H, Ta, Ts_peak] = ['nc_l', 'nc_w', 'nc_h', 'nc_ta', 'nc_ts'].map(id => parseFloat(document.getElementById(id).value));
            const material_props = materials[document.getElementById('nc_mat').value];
            const resultsDiv = document.getElementById('nc-results');
            if (Ts_peak <= Ta) { resultsDiv.innerHTML = `<p class="text-red-400">Error: Max. Surface Temp (Ts) must be higher than Ambient Temp (Ta).</p>`; return; }
            if (L <= 0 || W <= 0 || H <= 0) { resultsDiv.innerHTML = `<p class="text-red-400">Error: Product dimensions (L, W, H) must be greater than zero.</p>`; return; }
            const STEFAN_BOLTZMANN_CONST = 5.67e-8, EPSILON = 1e-9, BUILT_IN_SAFETY_FACTOR = 0.9, { emissivity, k_uniform } = material_props;
            const Ts_eff = Ta + (Ts_peak - Ta) * k_uniform, delta_T_eff = Ts_eff - Ta, [L_m, W_m, H_m] = [L/1000, W/1000, H/1000];
            const A_total = 2 * (L_m*W_m + L_m*H_m + W_m*H_m), T_film = (Ts_eff + Ta) / 2, [k_air, nu_air, pr_air, g] = [0.0275, 1.85e-5, 0.72, 9.81];
            const beta = 1 / (T_film + 273.15 + EPSILON), Lc_vert = H_m, Lc_horiz = (L_m * W_m) / (2 * (L_m + W_m) + EPSILON);
            const Ra_vert = (g*beta*delta_T_eff*Lc_vert**3)/(nu_air**2)*pr_air, Ra_horiz = (g*beta*delta_T_eff*Lc_horiz**3)/(nu_air**2)*pr_air;
            const Nu_vert = (0.825 + (0.387 * Math.abs(Ra_vert)**(1/6)) / (1 + (0.492/pr_air)**(9/16))**(8/27))**2;
            let Nu_top = 1.0; if (Ra_horiz >= 1e4 && Ra_horiz <= 1e7) Nu_top = 0.54 * Ra_horiz**(1/4); else if (Ra_horiz > 1e7) Nu_top = 0.15 * Ra_horiz**(1/3);
            const Nu_bottom = 0.27 * Math.abs(Ra_horiz)**(1/4);
            const h_vert = (Nu_vert*k_air)/(Lc_vert+EPSILON), h_top = (Nu_top*k_air)/(Lc_horiz+EPSILON), h_bottom = (Nu_bottom*k_air)/(Lc_horiz+EPSILON);
            const Q_conv_total = ((h_top*L_m*W_m)+(h_bottom*L_m*W_m)+h_vert*2*(L_m*H_m+W_m*H_m))*delta_T_eff;
            const Ts_eff_K = Ts_eff + 273.15, Ta_K = Ta + 273.15, Q_rad = emissivity * STEFAN_BOLTZMANN_CONST * A_total * (Ts_eff_K**4 - Ta_K**4);
            const Q_final = (Q_conv_total + Q_rad) * BUILT_IN_SAFETY_FACTOR;
            resultsDiv.innerHTML = `<div class="bg-slate-700/50 p-4 rounded-lg"><p class="text-sm text-slate-400">📐 Surface Area</p><p class="text-2xl font-bold">${A_total.toFixed(4)} m²</p></div><div class="bg-slate-700/50 p-4 rounded-lg"><p class="text-sm text-slate-400">✅ Max. Dissipatable Power</p><p class="text-2xl font-bold text-cyan-400">${Q_final.toFixed(2)} W</p><p class="text-xs text-slate-500 mt-1">Includes material uniformity and a fixed engineering safety factor (0.9).</p></div>`;
        }
        
        function calculateAndDisplayFC() {
            const [power_q, T_in, T_out] = ['fc_power_q', 'fc_tin', 'fc_tout'].map(id => parseFloat(document.getElementById(id).value));
            const resultsDiv = document.getElementById('fc-results');
            if (T_out <= T_in) { resultsDiv.innerHTML = `<p class="text-red-400">Error: Outlet Temperature must be higher than Inlet Temperature.</p>`; return; }
            if (power_q <= 0) { resultsDiv.innerHTML = `<p class="text-red-400">Error: Power to be dissipated must be greater than zero.</p>`; return; }
            const AIR_SPECIFIC_HEAT_CP = 1006, AIR_DENSITY_RHO = 1.225, M3S_TO_CFM_CONVERSION = 2118.88;
            const delta_T = T_out - T_in, mass_flow_rate = power_q / (AIR_SPECIFIC_HEAT_CP * delta_T);
            const volume_flow_rate_m3s = mass_flow_rate / AIR_DENSITY_RHO, cfm = volume_flow_rate_m3s * M3S_TO_CFM_CONVERSION;
            resultsDiv.innerHTML = `<div class="bg-slate-700/50 p-4 rounded-lg"><p class="text-sm text-slate-400">🌬️ Required Airflow</p><p class="text-2xl font-bold text-cyan-400">${cfm.toFixed(2)} CFM</p><p class="text-xs text-slate-500 mt-1">CFM: Cubic Feet per Minute.</p></div>`;
        }

        function calculateAndDisplaySolar() {
            const matSelect = document.getElementById('solar_mat'); let alpha = parseFloat(matSelect.value);
            if (matSelect.value === 'custom') alpha = parseFloat(document.getElementById('solar_alpha_custom').value) || 0;
            const [projected_area_mm2, solar_irradiance] = ['solar_area', 'solar_irradiance'].map(id => parseFloat(document.getElementById(id).value));
            const resultsDiv = document.getElementById('solar-results');
            if (projected_area_mm2 <= 0) { resultsDiv.innerHTML = `<p class="text-red-400">Error: Projected Surface Area must be greater than zero.</p>`; return; }
            const projected_area_m2 = projected_area_mm2 / 1_000_000, solar_gain = alpha * projected_area_m2 * solar_irradiance;
            resultsDiv.innerHTML = `<div class="bg-slate-700/50 p-4 rounded-lg"><p class="text-sm text-slate-400">☀️ Absorbed Solar Heat Gain</p><p class="text-2xl font-bold text-cyan-400">${solar_gain.toFixed(2)} W</p></div>`;
        }
        
        // --- COBRA JAVASCRIPT ---
        // Global state object for Cobra tool
        const cobraState = {
            prestudyData: null,
            analysisResults: null,
            fileBuffer: null,
            fileName: null,
            deltaPairs: [],
        };

        function renderCobraUI() {
            const cobraUI = document.getElementById('cobra-ui');
            const btnClasses = "w-full text-sm bg-slate-600 hover:bg-slate-500 text-white font-bold py-2 px-3 rounded-lg transition duration-200";
            const inputClasses = "w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-500";
            cobraUI.innerHTML = `
                <div class="flex items-center border-b-2 border-slate-700 pb-4 mb-6">
                    <div class="mr-4">
                        <svg width="50" height="50" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M40 95 C 20 95, 10 75, 25 60 C 35 50, 65 50, 75 60 C 90 75, 80 95, 60 95 L 40 95 Z" fill="#2C7873"/><path d="M50 70 C 25 70, 25 40, 50 20 C 75 40, 75 70, 50 70 Z" fill="#4A938E" stroke="#FFFFFF" stroke-width="2"/><path d="M50 20 C 40 30, 40 50, 50 60 C 60 50, 60 30, 50 20" fill="#1E1E1E"/><circle cx="46" cy="45" r="4" fill="red"/><circle cx="54" cy="45" r="4" fill="red"/><path d="M48 62 L52 62 L50 70 Z" fill="#FFD700"/></svg>
                    </div>
                    <div>
                        <h1 class="text-3xl font-bold text-white mb-0">Cobra</h1>
                        <p class="text-slate-400 mt-0">Data Transformation & Analysis</p>
                    </div>
                </div>

                <div class="grid grid-cols-12 gap-8">
                    <!-- Left Column: Controls -->
                    <div class="col-span-12 md:col-span-5">
                        <div class="space-y-6">
                            <div>
                                <h3 class="text-xl font-semibold mb-2">Upload Excel File</h3>
                                <div class="flex items-center space-x-2">
                                    <label class="${btnClasses} cursor-pointer text-center">
                                        <span id="cobra-file-label">Choose File</span>
                                        <input type="file" id="cobra-file-uploader" class="hidden" accept=".xlsx, .xls">
                                    </label>
                                    <button id="cobra-remove-file" class="${btnClasses} bg-red-800 hover:bg-red-700">Remove</button>
                                </div>
                            </div>
                            <div id="cobra-analysis-parameters" class="hidden">
                                <h3 class="text-xl font-semibold mb-2">Analysis Parameters</h3>
                                <div id="cobra-tabs" class="st-tabs-container">
                                    <button data-tab="series" class="st-tab active">1. Select Configurations</button>
                                    <button data-tab="ics" class="st-tab">2. Select Key ICs</button>
                                    <button data-tab="delta" class="st-tab">3. ΔT Comparison</button>
                                </div>
                                
                                <div id="cobra-series-content" class="st-tab-content active"></div>
                                <div id="cobra-ics-content" class="st-tab-content"></div>
                                <div id="cobra-delta-content" class="st-tab-content"></div>
                            </div>
                        </div>
                    </div>

                     <!-- Right Column: Specs & Analysis Button -->
                    <div class="col-span-12 md:col-span-7">
                        <div id="cobra-spec-editor-container" class="hidden">
                            <h3 class="text-xl font-semibold mb-2">4. Key IC Specification Input</h3>
                            <div id="cobra-spec-editor" class="overflow-auto max-h-96"></div>
                        </div>
                        <button id="cobra-analyze-btn" class="hidden mt-6 w-full bg-cyan-600 hover:bg-cyan-700 text-white font-bold py-3 px-4 rounded-lg transition duration-300 text-lg">🚀 Analyze Selected Data</button>
                    </div>
                </div>

                <!-- Results Area -->
                <div id="cobra-results-area" class="mt-8 hidden"></div>
            `;
            setupCobraEventListeners();
        }

        function setupCobraEventListeners() {
            // File uploader
            document.getElementById('cobra-file-uploader').addEventListener('change', handleCobraFile);
            document.getElementById('cobra-remove-file').addEventListener('click', resetCobraState);

            // Analyze button
            document.getElementById('cobra-analyze-btn').addEventListener('click', runCobraAnalysis);
        }

        function resetCobraState() {
            Object.assign(cobraState, { prestudyData: null, analysisResults: null, fileBuffer: null, fileName: null, deltaPairs: [] });
            document.getElementById('cobra-file-uploader').value = ''; // Clear file input
            document.getElementById('cobra-file-label').textContent = 'Choose File';
            document.getElementById('cobra-analysis-parameters').classList.add('hidden');
            document.getElementById('cobra-spec-editor-container').classList.add('hidden');
            document.getElementById('cobra-analyze-btn').classList.add('hidden');
            document.getElementById('cobra-results-area').classList.add('hidden');
        }
        
        function handleCobraFile(event) {
            const file = event.target.files[0];
            if (!file) return;

            resetCobraState();
            cobraState.fileName = file.name;
            document.getElementById('cobra-file-label').textContent = file.name;

            const reader = new FileReader();
            reader.onload = (e) => {
                cobraState.fileBuffer = e.target.result;
                cobraPreStudy();
            };
            reader.readAsArrayBuffer(file);
        }

        // Cobra Logic (to be filled)
        function cobraPreStudy() {
            if (!cobraState.fileBuffer) return;
            // ... Logic to parse the file and populate cobraState.prestudyData
            // ... Then render the parameter selection UI
            console.log("Pre-study would run here.");
        }
        function runCobraAnalysis() {
            // ... Logic to get user selections
            // ... Run the main analysis
            // ... Populate cobraState.analysisResults
            // ... Render results UI
             console.log("Analysis would run here.");
        }


        // Run the application
        initializeAndRunApp();

    </script>
</body>
</html>
