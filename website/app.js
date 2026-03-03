/* ============================================================
   DAIN AI — Application Logic
   Drug Simulation Engine + UI Management + Chart.js Visualization
   ============================================================ */

// ============================================================
// DRUG DATABASE
// ============================================================
const DRUG_DATABASE = [
    {
        name: 'Aspirin (Acetylsalicylic Acid)',
        smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O',
        drugClass: 'NSAID / Antiplatelet',
        halfLifeMin: 20,
        defaultDoseMg: 325,
        volumeL: 10,
        primaryEffect: { lab: 'LBXCRP', EC50: 20, Emax: 0.8, direction: 'decrease' },
        secondaryEffects: {
            LBXSGT: { EC50: 25, Emax: 5, direction: 'increase' },
        }
    },
    {
        name: 'Ibuprofen',
        smiles: 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',
        drugClass: 'NSAID',
        halfLifeMin: 120,
        defaultDoseMg: 400,
        volumeL: 10,
        primaryEffect: { lab: 'LBXCRP', EC50: 25, Emax: 1.0, direction: 'decrease' },
        secondaryEffects: {
            LBDSCR: { EC50: 30, Emax: 0.15, direction: 'increase' },
        }
    },
    {
        name: 'Regular Insulin',
        smiles: 'C(CC(C(=O)O)N)CN=C(N)N',
        drugClass: 'Antidiabetic',
        halfLifeMin: 5,
        defaultDoseMg: 10,
        volumeL: 15,
        primaryEffect: { lab: 'LBXSGL', EC50: 0.5, Emax: 85, direction: 'decrease' },
        secondaryEffects: {
            LBXSK: { EC50: 0.5, Emax: 0.5, direction: 'decrease' },
        }
    },
    {
        name: 'Furosemide',
        smiles: 'C1=CC(=CC(=C1)S(=O)(=O)N)NC2=CC=CC=C2Cl',
        drugClass: 'Loop Diuretic',
        halfLifeMin: 75,
        defaultDoseMg: 40,
        volumeL: 15,
        primaryEffect: { lab: 'LBXSK', EC50: 2, Emax: 0.6, direction: 'decrease' },
        secondaryEffects: {
            LBXSNA: { EC50: 2, Emax: 3, direction: 'decrease' },
            LBXSCA: { EC50: 2, Emax: 0.3, direction: 'increase' },
            LBDSCR: { EC50: 2, Emax: 0.2, direction: 'increase' },
            LBXSBU: { EC50: 2, Emax: 4, direction: 'increase' },
        }
    },
    {
        name: 'Potassium Chloride',
        smiles: '[K+].[Cl-]',
        drugClass: 'Electrolyte Supplement',
        halfLifeMin: 120,
        defaultDoseMg: 1560,
        volumeL: 40,
        primaryEffect: { lab: 'LBXSK', EC50: 10, Emax: 1.5, direction: 'increase' },
        secondaryEffects: {}
    },
    {
        name: 'Epinephrine',
        smiles: 'CNCC(C1=CC(=C(C=C1)O)O)O',
        drugClass: 'Sympathomimetic',
        halfLifeMin: 2.5,
        defaultDoseMg: 0.5,
        volumeL: 60,
        primaryEffect: { lab: 'LBXSGL', EC50: 0.01, Emax: 30, direction: 'increase' },
        secondaryEffects: {
            LBXSK: { EC50: 0.01, Emax: 0.3, direction: 'increase' },
        }
    },
    {
        name: 'Metformin',
        smiles: 'CN(C)C(=N)NC(=N)N',
        drugClass: 'Antidiabetic (Biguanide)',
        halfLifeMin: 300,
        defaultDoseMg: 500,
        volumeL: 65,
        primaryEffect: { lab: 'LBXSGL', EC50: 5, Emax: 40, direction: 'decrease' },
        secondaryEffects: {
            LBXTC: { EC50: 6, Emax: 15, direction: 'decrease' },
            LBDSCR: { EC50: 8, Emax: 0.1, direction: 'increase' },
        }
    },
    {
        name: 'Atorvastatin',
        smiles: 'CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)[O-])O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4',
        drugClass: 'Statin (HMG-CoA Reductase Inhibitor)',
        halfLifeMin: 840,
        defaultDoseMg: 20,
        volumeL: 380,
        primaryEffect: { lab: 'LBXTC', EC50: 0.03, Emax: 60, direction: 'decrease' },
        secondaryEffects: {
            LBDSCH: { EC50: 0.03, Emax: 10, direction: 'increase' },
            LBXSAS: { EC50: 0.02, Emax: 8, direction: 'increase' },
            LBXSGT: { EC50: 0.02, Emax: 6, direction: 'increase' },
        }
    },
    {
        name: 'Lisinopril',
        smiles: 'C(CC(C(=O)O)NC(CCC1=CC=CC=C1)C(=O)O)CN',
        drugClass: 'ACE Inhibitor',
        halfLifeMin: 720,
        defaultDoseMg: 10,
        volumeL: 120,
        primaryEffect: { lab: 'LBXSK', EC50: 0.05, Emax: 0.4, direction: 'increase' },
        secondaryEffects: {
            LBDSCR: { EC50: 0.06, Emax: 0.15, direction: 'increase' },
        }
    },
    {
        name: 'Paracetamol (Acetaminophen)',
        smiles: 'CC(=O)NC1=CC=C(C=C1)O',
        drugClass: 'Analgesic / Antipyretic',
        halfLifeMin: 150,
        defaultDoseMg: 500,
        volumeL: 50,
        primaryEffect: { lab: 'LBXCRP', EC50: 8, Emax: 0.6, direction: 'decrease' },
        secondaryEffects: {
            LBXSAS: { EC50: 10, Emax: 6, direction: 'increase' },
            LBXSGT: { EC50: 10, Emax: 5, direction: 'increase' },
        }
    },
];

// ============================================================
// LAB BIOMARKER DEFINITIONS
// ============================================================
const LAB_DEFINITIONS = {
    LBDSCASI: { name: 'Sedation Level Score', unit: '', normal: [0, 3] },
    LBDSCH: { name: 'HDL Cholesterol', unit: 'mg/dL', normal: [40, 60], betterDir: 'increase' },
    LBDSCR: { name: 'Creatinine (kidney)', unit: 'mg/dL', normal: [0.6, 1.2] },
    LBDTC: { name: 'Total Cholesterol (derived)', unit: 'mg/dL', normal: [80, 200] },
    LBXBCD: { name: 'Blood Cadmium', unit: 'µg/L', normal: [0.1, 1.0] },
    LBXBPB: { name: 'Blood Lead', unit: 'µg/dL', normal: [0, 5] },
    LBXCRP: { name: 'C-Reactive Protein (inflammation)', unit: 'mg/L', normal: [0, 3], betterDir: 'decrease' },
    LBXSAL: { name: 'Albumin (blood protein)', unit: 'g/dL', normal: [3.5, 5.0] },
    LBXSAS: { name: 'AST (liver enzyme)', unit: 'U/L', normal: [10, 40] },
    LBXSBU: { name: 'Blood Urea Nitrogen', unit: 'mg/dL', normal: [7, 20] },
    LBXSCA: { name: 'Calcium', unit: 'mg/dL', normal: [8.5, 10.5] },
    LBXSCH: { name: 'Total Cholesterol', unit: 'mg/dL', normal: [125, 200] },
    LBXSCL: { name: 'Chloride', unit: 'mmol/L', normal: [98, 106] },
    LBXSGB: { name: 'Globulin', unit: 'g/dL', normal: [2.0, 3.5] },
    LBXSGL: { name: 'Blood Glucose', unit: 'mg/dL', normal: [70, 100] },
    LBXSGT: { name: 'GGT (liver enzyme)', unit: 'U/L', normal: [8, 61] },
    LBXSK: { name: 'Potassium', unit: 'mmol/L', normal: [3.5, 5.0] },
    LBXSNA: { name: 'Sodium', unit: 'mmol/L', normal: [136, 145] },
    LBXSOS: { name: 'Blood Osmolality', unit: 'mOsm/kg', normal: [275, 295] },
    LBXSTP: { name: 'Total Protein', unit: 'g/dL', normal: [6.0, 8.3] },
    LBXSUA: { name: 'Uric Acid', unit: 'mg/dL', normal: [3.5, 7.2] },
    LBXTC: { name: 'Total Cholesterol (lab)', unit: 'mg/dL', normal: [125, 200] },
};

// Age groups for cohort simulations
const AGE_GROUPS = [
    { id: '18-30', label: '18–30', min: 18, max: 30 },
    { id: '31-45', label: '31–45', min: 31, max: 45 },
    { id: '46-60', label: '46–60', min: 46, max: 60 },
    { id: '61-85', label: '61–85', min: 61, max: 85 },
];

// ============================================================
// PHARMACOKINETIC ENGINE (ported from test.py)
// ============================================================

/**
 * One-compartment IV bolus PK model — drug concentration at time t.
 */
function drugConcentration(timeSec, doseMg, volumeL, halfLifeMin) {
    if (timeSec < 0) return 0;
    const tMin = timeSec / 60;
    const kElim = 0.693 / halfLifeMin;
    const C0 = doseMg / volumeL;
    return Math.max(0, C0 * Math.exp(-kElim * tMin));
}

/**
 * Emax pharmacodynamic model — calculate drug effect on lab value.
 */
function calculateEffect(drugConc, EC50, Emax, baseline, direction) {
    if (drugConc <= 0) return baseline;
    const effect = (Emax * drugConc) / (EC50 + drugConc);
    return direction === 'decrease' ? baseline - effect : baseline + effect;
}

/**
 * Add realistic Gaussian measurement noise.
 */
function addMeasurementNoise(value, cvPercent = 3) {
    const noise = gaussianRandom() * (cvPercent / 100);
    return value * (1 + noise);
}

/**
 * Add biological variability (circadian + short-term oscillation).
 */
function addBiologicalVariability(value, timeSec, cvPercent = 5) {
    const tMin = timeSec / 60;
    const circadian = Math.sin(2 * Math.PI * tMin / (24 * 60)) * (cvPercent / 200);
    const shortTerm = Math.sin(2 * Math.PI * tMin / 5) * (cvPercent / 400);
    const random = gaussianRandom() * (cvPercent / 200);
    return value * (1 + circadian + shortTerm + random);
}

/**
 * Box-Muller transform for Gaussian random numbers.
 */
function gaussianRandom() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/**
 * Generate demographics-adjusted baseline lab values.
 */
function generateBaselineLabs(age, sex, drugName) {
    const labs = {};
    const isMale = sex === 'M';

    // Glucose — hyperglycemic for insulin test
    if (drugName && drugName.includes('Insulin')) {
        labs.LBXSGL = randUniform(160, 200);
    } else {
        labs.LBXSGL = randUniform(85, 110);
    }

    // Electrolytes
    labs.LBXSK = randUniform(3.8, 4.5);
    labs.LBXSNA = randUniform(138, 142);
    labs.LBXSCA = randUniform(9.0, 9.8);
    labs.LBXSOS = randUniform(280, 290);

    // Lipids
    labs.LBXTC = randUniform(160, 220);
    labs.LBXSCH = randUniform(160, 220);
    labs.LBXSCL = randUniform(98, 104);
    labs.LBDSCH = isMale ? randUniform(40, 60) : randUniform(50, 70);
    labs.LBDTC = randUniform(80, 180);

    // Liver function
    labs.LBXSGT = randUniform(15, 40);
    labs.LBXSAS = randUniform(15, 35);
    labs.LBXSGB = randUniform(2.2, 3.3);
    labs.LBXSAL = randUniform(3.8, 4.5);
    labs.LBXSTP = randUniform(6.5, 7.8);

    // Kidney function
    labs.LBDSCR = isMale ? randUniform(0.8, 1.2) : randUniform(0.6, 1.0);
    labs.LBXSUA = isMale ? randUniform(4.5, 6.5) : randUniform(3.5, 5.5);
    labs.LBXSBU = randUniform(10, 18);

    // Other
    labs.LBXCRP = randUniform(0.5, 2.0);
    labs.LBXBCD = randUniform(0.2, 1.0);
    labs.LBXBPB = randUniform(0.8, 3.0);
    labs.LBDSCASI = randUniform(0, 2);

    // Age adjustments
    if (age > 60) {
        labs.LBDSCR *= 1.1;
        labs.LBXSBU *= 1.1;
        labs.LBXTC *= 1.05;
        labs.LBXCRP *= 1.2;
    }

    return labs;
}

function randUniform(min, max) {
    return min + Math.random() * (max - min);
}

/**
 * Run complete time series simulation for one patient + one drug.
 * Returns: { times: [...], series: { labCode: [...values] }, baseline: {...}, drug_conc: [...] }
 */
function runSimulation(patient, drugConfig, doseMg, durationSec, intervalSec = 30) {
    const baselines = generateBaselineLabs(patient.age, patient.sex, drugConfig.name);
    const times = [];
    const drugConcs = [];
    const series = {};

    // Initialize series arrays for every lab
    for (const lab of Object.keys(baselines)) {
        series[lab] = [];
    }

    const numSteps = Math.floor(durationSec / intervalSec) + 1;

    for (let i = 0; i < numSteps; i++) {
        const t = i * intervalSec;
        times.push(t);

        // Drug concentration
        const conc = drugConcentration(t, doseMg, drugConfig.volumeL, drugConfig.halfLifeMin);
        drugConcs.push(conc);

        // For each lab, compute value at time t
        for (const [labCode, baselineVal] of Object.entries(baselines)) {
            let value = baselineVal;

            // Check if this lab is affected by the drug
            if (drugConfig.primaryEffect && drugConfig.primaryEffect.lab === labCode) {
                const pe = drugConfig.primaryEffect;
                value = calculateEffect(conc, pe.EC50, pe.Emax, baselineVal, pe.direction);
                value = addBiologicalVariability(value, t, 5);
                value = addMeasurementNoise(value, 3);
            } else if (drugConfig.secondaryEffects && drugConfig.secondaryEffects[labCode]) {
                const se = drugConfig.secondaryEffects[labCode];
                value = calculateEffect(conc, se.EC50, se.Emax, baselineVal, se.direction);
                value = addBiologicalVariability(value, t, 3);
                value = addMeasurementNoise(value, 2);
            } else {
                // Unaffected labs: tiny drift
                const drift = ['LBXTC', 'LBXSCL', 'LBDSCH', 'LBDTC', 'LBXSGT', 'LBXSAS', 'LBXBCD', 'LBXBPB'];
                const electrolytes = ['LBXSNA', 'LBXSCA', 'LBXSOS'];
                if (drift.includes(labCode)) {
                    value = baselineVal * (1 + gaussianRandom() * 0.001);
                } else if (electrolytes.includes(labCode)) {
                    value = baselineVal * (1 + gaussianRandom() * 0.002);
                } else {
                    value = baselineVal * (1 + gaussianRandom() * 0.005);
                }
            }

            series[labCode].push(Math.max(0, value));
        }
    }

    return { times, series, baselines, drugConcs };
}

/**
 * Run cohort simulation for multiple synthetic patients.
 * Returns aggregated stats and per-patient classifications.
 */
function runCohortSimulation(patientCount, drug, dosage, durationSec) {
    const patients = generateCohortPatients(patientCount);
    const perPatient = [];

    const labStats = {};

    function ensureLabStats(labCode) {
        if (!labStats[labCode]) {
            const def = LAB_DEFINITIONS[labCode] || {};
            labStats[labCode] = {
                labCode,
                name: def.name || labCode,
                unit: def.unit || '',
                normal: def.normal || null,
                physiological: 0,
                pathological: 0,
                total: 0,
                bySex: {
                    M: { pathological: 0, total: 0 },
                    F: { pathological: 0, total: 0 },
                },
                byAgeGroup: AGE_GROUPS.reduce((acc, g) => {
                    acc[g.id] = { pathological: 0, total: 0 };
                    return acc;
                }, {}),
            };
        }
        return labStats[labCode];
    }

    patients.forEach((patient) => {
        const sim = runSimulation(patient, drug, dosage, durationSec, 30);
        const classification = classifyMetricsForSimulation(sim.times, sim.series, drug);

        const unstable = classification.metrics.some(m => m.status === 'worsened');
        perPatient.push({ patient, sim, classification, unstable });

        classification.metrics.forEach((m) => {
            const stat = ensureLabStats(m.labCode);
            const isPathological = m.status === 'worsened';
            if (isPathological) {
                stat.pathological += 1;
            } else {
                stat.physiological += 1;
            }
            stat.total += 1;

            const sexBucket = stat.bySex[patient.sex];
            if (sexBucket) {
                sexBucket.total += 1;
                if (isPathological) sexBucket.pathological += 1;
            }

            const ageGroup = patient.ageGroup && stat.byAgeGroup[patient.ageGroup.id];
            if (ageGroup) {
                ageGroup.total += 1;
                if (isPathological) ageGroup.pathological += 1;
            }
        });
    });

    const labStatsArray = Object.values(labStats);

    const totalLabsPerPatient = labStatsArray.length;
    const totalObservations = totalLabsPerPatient * patientCount;
    const totalPathological = labStatsArray.reduce((sum, l) => sum + l.pathological, 0);
    const totalPhysiological = totalObservations - totalPathological;

    const ageInsights = AGE_GROUPS.map(group => {
        const perLab = labStatsArray.map(l => {
            const bucket = l.byAgeGroup[group.id];
            const rate = bucket && bucket.total ? bucket.pathological / bucket.total : 0;
            return { labCode: l.labCode, name: l.name, rate };
        }).sort((a, b) => b.rate - a.rate);
        const top = perLab[0] || null;
        return { group, top };
    });

    const sexInsights = ['M', 'F'].map(sex => {
        const perLab = labStatsArray.map(l => {
            const bucket = l.bySex[sex];
            const rate = bucket && bucket.total ? bucket.pathological / bucket.total : 0;
            return { labCode: l.labCode, name: l.name, rate };
        }).sort((a, b) => b.rate - a.rate);
        const top = perLab[0] || null;
        return { sex, top };
    });

    const analysisCohortSize = Math.min(patientCount, 40);
    const analysisPatients = patients.slice(0, analysisCohortSize);
    const drugInstability = DRUG_DATABASE.map(d => {
        let unstablePatients = 0;
        analysisPatients.forEach(p => {
            const sim = runSimulation(p, d, d.defaultDoseMg, durationSec, 60);
            const classification = classifyMetricsForSimulation(sim.times, sim.series, d);
            const unstable = classification.metrics.some(m => m.status === 'worsened');
            if (unstable) unstablePatients += 1;
        });
        const rate = analysisCohortSize ? unstablePatients / analysisCohortSize : 0;
        return { drug: d, unstablePatients, totalPatients: analysisCohortSize, rate };
    }).sort((a, b) => b.rate - a.rate);

    return {
        patients,
        perPatient,
        labStats: labStatsArray,
        summary: {
            patientCount,
            totalLabsPerPatient,
            totalObservations,
            totalPathological,
            totalPhysiological,
        },
        ageInsights,
        sexInsights,
        drugInstability,
        baseDrug: drug,
        dosage,
        durationSec,
    };
}

function generateCohortPatients(patientCount) {
    const patients = [];
    const sexes = [];
    const half = Math.floor(patientCount / 2);
    for (let i = 0; i < patientCount; i++) {
        sexes.push(i < half ? 'M' : 'F');
    }
    for (let i = sexes.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [sexes[i], sexes[j]] = [sexes[j], sexes[i]];
    }

    for (let i = 0; i < patientCount; i++) {
        const sex = sexes[i];
        const group = AGE_GROUPS[i % AGE_GROUPS.length];
        const age = Math.floor(randUniform(group.min, group.max + 0.99));

        const isMale = sex === 'M';
        const height = isMale ? randUniform(165, 195) : randUniform(155, 180);
        const weight = isMale ? randUniform(60, 110) : randUniform(50, 95);
        const bmi = weight / Math.pow(height / 100, 2);

        patients.push({
            age,
            sex,
            height,
            weight,
            bmi,
            ageGroup: group,
        });
    }

    return patients;
}

// ============================================================
// UI STATE & MANAGEMENT
// ============================================================
let selectedDrug = null;
let simulationResult = null;
let chartInstances = [];
let simulationMode = 'single'; // 'single' | 'cohort'
let cohortResult = null;

document.addEventListener('DOMContentLoaded', () => {
    initParticles();
    initDrugSelector();
    initFormControls();
    initFormSubmission();
    initFilterTabs();

    document.getElementById('btn-new-sim').addEventListener('click', goToInput);
    const downloadBtn = document.getElementById('btn-download-zip');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', downloadSimulationZip);
    }
});

// ============================================================
// DRUG SELECTOR
// ============================================================
function initDrugSelector() {
    const select = document.getElementById('drug-select');
    DRUG_DATABASE.forEach((drug, idx) => {
        const opt = document.createElement('option');
        opt.value = idx;
        opt.textContent = drug.name;
        select.appendChild(opt);
    });

    select.addEventListener('change', (e) => {
        if (e.target.value === '') {
            selectedDrug = null;
            document.getElementById('drug-info-card').style.display = 'none';
            updateSmilesPreview('');
            return;
        }
        const drug = DRUG_DATABASE[parseInt(e.target.value)];
        selectedDrug = drug;
        document.getElementById('drug-info-class').textContent = drug.drugClass;
        document.getElementById('drug-info-halflife').textContent = formatHalfLife(drug.halfLifeMin);
        document.getElementById('drug-info-effect').textContent = `${drug.primaryEffect.direction === 'decrease' ? '↓' : '↑'} ${LAB_DEFINITIONS[drug.primaryEffect.lab]?.name || drug.primaryEffect.lab}`;
        document.getElementById('drug-info-card').style.display = 'block';
        updateSmilesPreview(drug.smiles);

        // Update dosage slider to drug's default
        const dosageSlider = document.getElementById('drug-dosage');
        dosageSlider.value = Math.min(200, drug.defaultDoseMg);
        document.getElementById('dosage-display').textContent = `${dosageSlider.value} mg`;
    });

    // Source toggle
    const dbBtn = document.getElementById('source-database');
    const customBtn = document.getElementById('source-custom');

    dbBtn.addEventListener('click', () => {
        dbBtn.classList.add('active');
        customBtn.classList.remove('active');
        document.getElementById('database-drug-group').style.display = '';
        document.getElementById('drug-info-card').style.display = selectedDrug ? 'block' : 'none';
        document.getElementById('custom-smiles-group').style.display = 'none';
        if (selectedDrug) updateSmilesPreview(selectedDrug.smiles);
    });

    customBtn.addEventListener('click', () => {
        customBtn.classList.add('active');
        dbBtn.classList.remove('active');
        document.getElementById('database-drug-group').style.display = 'none';
        document.getElementById('drug-info-card').style.display = 'none';
        document.getElementById('custom-smiles-group').style.display = '';
        updateSmilesPreview(document.getElementById('smiles-input').value);
    });

    document.getElementById('smiles-input').addEventListener('input', (e) => {
        updateSmilesPreview(e.target.value);
    });
}

function formatHalfLife(minutes) {
    if (minutes < 60) return `${minutes} min`;
    if (minutes < 1440) return `${(minutes / 60).toFixed(1)} hours`;
    return `${(minutes / 1440).toFixed(1)} days`;
}

function updateSmilesPreview(smiles) {
    const preview = document.getElementById('smiles-preview');
    if (smiles && smiles.trim()) {
        preview.innerHTML = `<code>${escapeHtml(smiles)}</code>`;
    } else {
        preview.innerHTML = `<code style="color:var(--text-muted)">Select a drug or enter SMILES</code>`;
    }
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ============================================================
// FORM CONTROLS
// ============================================================
function initFormControls() {
    // Sex toggle
    document.querySelectorAll('.sex-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.sex-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById('patient-sex').value = btn.dataset.value;
            updateBMI();
        });
    });

    // Sliders
    const sliders = [
        { id: 'patient-age', display: 'age-display', suffix: ' years', inputId: 'patient-age-input', type: 'int' },
        { id: 'patient-height', display: 'height-display', suffix: ' cm', inputId: 'patient-height-input', type: 'int' },
        { id: 'patient-weight', display: 'weight-display', suffix: ' kg', inputId: 'patient-weight-input', type: 'int' },
        { id: 'patient-count', display: 'patient-count-display', suffix: ' patient', pluralSuffix: ' patients', inputId: 'patient-count-input', type: 'int' },
        { id: 'drug-dosage', display: 'dosage-display', suffix: ' mg', inputId: 'drug-dosage-input', type: 'float' },
        { id: 'sim-duration', display: 'duration-display', suffix: ' hours', inputId: 'sim-duration-input', type: 'int' },
    ];

    sliders.forEach(({ id, display, suffix, pluralSuffix, inputId, type }) => {
        const slider = document.getElementById(id);
        const disp = document.getElementById(display);
        const numeric = document.getElementById(inputId);

        if (!slider || !disp || !numeric) return;

        const min = slider.min !== '' ? Number(slider.min) : undefined;
        const max = slider.max !== '' ? Number(slider.max) : undefined;

        const syncFromSlider = () => {
            const raw = type === 'float' ? parseFloat(slider.value) : parseInt(slider.value, 10);
            const val = isNaN(raw) ? (type === 'float' ? 0 : 0) : raw;
            numeric.value = val;

            if (id === 'patient-count') {
                const labelSuffix = val === 1 ? suffix : (pluralSuffix || ' patients');
                disp.textContent = `${val} ${labelSuffix}`;
                setCohortModeUI(val > 1);
            } else {
                disp.textContent = slider.value + suffix;
            }

            if (id === 'patient-height' || id === 'patient-weight') updateBMI();
        };

        const syncFromNumeric = () => {
            let raw = type === 'float' ? parseFloat(numeric.value) : parseInt(numeric.value, 10);
            if (isNaN(raw)) raw = Number(slider.value);
            if (min !== undefined && raw < min) raw = min;
            if (max !== undefined && raw > max) raw = max;

            slider.value = String(raw);
            if (id === 'patient-count') {
                const labelSuffix = raw === 1 ? suffix : (pluralSuffix || ' patients');
                disp.textContent = `${raw} ${labelSuffix}`;
            } else {
                disp.textContent = raw + suffix;
            }

            if (id === 'patient-height' || id === 'patient-weight') updateBMI();
        };

        slider.addEventListener('input', syncFromSlider);
        numeric.addEventListener('change', syncFromNumeric);
        numeric.addEventListener('blur', syncFromNumeric);

        syncFromSlider();
    });

    updateBMI();
}

function setCohortModeUI(isCohort) {
    const sexButtons = document.querySelectorAll('.sex-btn');
    const ageSlider = document.getElementById('patient-age');
    const heightSlider = document.getElementById('patient-height');
    const weightSlider = document.getElementById('patient-weight');
    const ageInput = document.getElementById('patient-age-input');
    const heightInput = document.getElementById('patient-height-input');
    const weightInput = document.getElementById('patient-weight-input');

    const ageDisplay = document.getElementById('age-display');
    const heightDisplay = document.getElementById('height-display');
    const weightDisplay = document.getElementById('weight-display');

    if (isCohort) {
        sexButtons.forEach(btn => {
            btn.classList.add('cohort-disabled');
            btn.disabled = true;
        });

        if (ageSlider) ageSlider.disabled = true;
        if (heightSlider) heightSlider.disabled = true;
        if (weightSlider) weightSlider.disabled = true;
        if (ageInput) ageInput.disabled = true;
        if (heightInput) heightInput.disabled = true;
        if (weightInput) weightInput.disabled = true;

        if (ageDisplay) ageDisplay.textContent = 'Randomized';
        if (heightDisplay) heightDisplay.textContent = 'Randomized';
        if (weightDisplay) weightDisplay.textContent = 'Randomized';
    } else {
        sexButtons.forEach(btn => {
            btn.classList.remove('cohort-disabled');
            btn.disabled = false;
        });

        if (ageSlider) ageSlider.disabled = false;
        if (heightSlider) heightSlider.disabled = false;
        if (weightSlider) weightSlider.disabled = false;
        if (ageInput) ageInput.disabled = false;
        if (heightInput) heightInput.disabled = false;
        if (weightInput) weightInput.disabled = false;

        // Re-sync displays from current slider values
        ['patient-age', 'patient-height', 'patient-weight', 'patient-count'].forEach(id => {
            const slider = document.getElementById(id);
            if (slider) {
                slider.dispatchEvent(new Event('input'));
            }
        });
    }
}

function updateBMI() {
    const height = parseFloat(document.getElementById('patient-height').value);
    const weight = parseFloat(document.getElementById('patient-weight').value);
    const bmi = weight / Math.pow(height / 100, 2);

    document.getElementById('bmi-number').textContent = bmi.toFixed(1);

    let category;
    if (bmi < 18.5) category = 'Underweight';
    else if (bmi < 25) category = 'Normal';
    else if (bmi < 30) category = 'Overweight';
    else category = 'Obese';

    document.getElementById('bmi-category').textContent = category;
}

// ============================================================
// FORM SUBMISSION & SIMULATION
// ============================================================
function initFormSubmission() {
    document.getElementById('simulation-form').addEventListener('submit', async (e) => {
        e.preventDefault();

        // Gather inputs
        const patientCount = parseInt(document.getElementById('patient-count').value, 10) || 1;
        const sex = document.getElementById('patient-sex').value;
        const age = parseInt(document.getElementById('patient-age').value);
        const height = parseFloat(document.getElementById('patient-height').value);
        const weight = parseFloat(document.getElementById('patient-weight').value);
        const dosage = parseFloat(document.getElementById('drug-dosage').value);
        const durationHours = parseInt(document.getElementById('sim-duration').value);
        const durationSec = durationHours * 3600;

        // Get drug config
        const isCustom = document.getElementById('source-custom').classList.contains('active');
        let drug;
        let smiles;

        if (isCustom) {
            smiles = document.getElementById('smiles-input').value.trim();
            if (!smiles) {
                showToast('Please enter a valid SMILES string.', 'error');
                return;
            }
            // Create a generic drug config for custom SMILES
            drug = createGenericDrugConfig(smiles);
        } else {
            if (!selectedDrug) {
                showToast('Please select a drug from the database.', 'error');
                return;
            }
            drug = selectedDrug;
            smiles = drug.smiles;
        }

        const patient = { age, sex, height, weight, bmi: weight / Math.pow(height / 100, 2) };

        // Switch to loading screen
        await switchScreen('screen-input', 'screen-loading');

        // Run loading animation with simulation
        await runLoadingAnimation(patient, drug, dosage, durationSec, patientCount);
    });
}

/**
 * Create a generic drug config for custom SMILES (assigns random but reasonable PK params).
 */
function createGenericDrugConfig(smiles) {
    // Estimate molecular complexity from SMILES length to adjust PK
    const complexity = Math.min(smiles.length / 10, 5);
    return {
        name: 'Custom Compound',
        smiles: smiles,
        drugClass: 'Unknown',
        halfLifeMin: 60 + complexity * 30,
        defaultDoseMg: 100,
        volumeL: 20 + complexity * 8,
        primaryEffect: { lab: 'LBXSGL', EC50: 5, Emax: 25, direction: 'decrease' },
        secondaryEffects: {
            LBXCRP: { EC50: 6, Emax: 0.5, direction: 'decrease' },
            LBXSAS: { EC50: 8, Emax: 4, direction: 'increase' },
        }
    };
}

// ============================================================
// LOADING ANIMATION
// ============================================================
async function runLoadingAnimation(patient, drug, dosage, durationSec, patientCount) {
    const steps = document.querySelectorAll('.progress-step');
    const progressBar = document.getElementById('progress-bar');
    const loadingTitle = document.getElementById('loading-title');
    const loadingSub = document.getElementById('loading-subtitle');

    // Reset
    steps.forEach(s => { s.classList.remove('active', 'done'); });
    progressBar.style.width = '0%';

    const isCohort = patientCount > 1;

    const stepData = [
        {
            title: isCohort ? 'Generating Synthetic Cohort' : 'Generating Synthetic Patient',
            sub: isCohort
                ? `Creating ${patientCount} diverse virtual patients…`
                : `Creating ${patient.age}yo ${patient.sex === 'M' ? 'male' : 'female'} profile…`,
            duration: 600
        },
        { title: 'Encoding Drug Molecule', sub: `Processing SMILES: ${drug.smiles.substring(0, 40)}…`, duration: 500 },
        { title: 'Computing PK/PD Response', sub: 'Running pharmacokinetic models…', duration: 800 },
        { title: 'Building Time Series', sub: `Generating ${Math.floor(durationSec / 30)} data points…`, duration: 700 },
    ];

    for (let i = 0; i < stepData.length; i++) {
        steps[i].classList.add('active');
        loadingTitle.textContent = stepData[i].title;
        loadingSub.textContent = stepData[i].sub;
        progressBar.style.width = `${((i + 0.5) / stepData.length) * 100}%`;

        // Actually run simulation on last step
        if (i === stepData.length - 1) {
            await sleep(200);
            if (isCohort) {
                simulationMode = 'cohort';
                cohortResult = runCohortSimulation(patientCount, drug, dosage, durationSec);
                simulationResult = null;
            } else {
                simulationMode = 'single';
                simulationResult = runSimulation(patient, drug, dosage, durationSec, 30);
                simulationResult.patient = patient;
                simulationResult.drug = drug;
                simulationResult.dosage = dosage;
                simulationResult.durationSec = durationSec;
                cohortResult = null;
            }
        }

        await sleep(stepData[i].duration);
        steps[i].classList.remove('active');
        steps[i].classList.add('done');
        progressBar.style.width = `${((i + 1) / stepData.length) * 100}%`;
    }

    await sleep(400);
    renderResults();
    await switchScreen('screen-loading', 'screen-results');
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================================
// SCREEN TRANSITIONS
// ============================================================
async function switchScreen(fromId, toId) {
    const fromScreen = document.getElementById(fromId);
    const toScreen = document.getElementById(toId);

    fromScreen.classList.add('fade-out');
    await sleep(300);
    fromScreen.classList.remove('active', 'fade-out');

    toScreen.classList.add('active');
}

function goToInput() {
    // Destroy existing charts
    chartInstances.forEach(c => c.destroy());
    chartInstances = [];

    simulationMode = 'single';
    simulationResult = null;
    cohortResult = null;

    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active', 'fade-out'));
    document.getElementById('screen-input').classList.add('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============================================================
// RESULTS RENDERING
// ============================================================
function renderResults() {
    // Hide cohort insights by default; specific renderers will toggle as needed
    const cohortEl = document.getElementById('cohort-insights');
    if (cohortEl) cohortEl.style.display = 'none';

    if (simulationMode === 'cohort' && cohortResult) {
        renderCohortResults();
    } else if (simulationResult) {
        renderSingleResult();
    }
}

function classifyMetricsForSimulation(times, series, drug) {
    const metrics = [];
    let improvedCount = 0;
    let worsenedCount = 0;

    for (const [labCode, values] of Object.entries(series)) {
        const def = LAB_DEFINITIONS[labCode];
        if (!def) continue;

        const baseline = values[0];
        const finalVal = values[values.length - 1];
        const delta = finalVal - baseline;
        const deltaPercent = baseline !== 0 ? (delta / baseline) * 100 : 0;

        let status = 'stable';
        let statusReason = '';

        const threshold = 2; // % change threshold to ignore tiny noise

        const hasNormal = !!def.normal;
        const belowNormal = hasNormal ? finalVal < def.normal[0] : false;
        const aboveNormal = hasNormal ? finalVal > def.normal[1] : false;
        const outOfNormal = belowNormal || aboveNormal;

        const isPrimary = drug.primaryEffect?.lab === labCode;
        const isSecondary = drug.secondaryEffects?.[labCode] !== undefined;
        const isDrugTarget = isPrimary || isSecondary;

        // 1) True pathological = outside normal range (dangerous)
        if (outOfNormal) {
            status = 'worsened';
            statusReason = belowNormal ? 'Below normal' : 'Above normal';
        } else {
            // 2) Inside normal range: only distinguish improved vs stable based on direction
            const bigEnoughChange = Math.abs(deltaPercent) > threshold;

            if (isDrugTarget && bigEnoughChange) {
                const effectConfig = isPrimary ? drug.primaryEffect : drug.secondaryEffects[labCode];
                const expectedDir = effectConfig.direction;

                const movedDown = delta < 0;
                const movedUp = delta > 0;

                if (expectedDir === 'decrease' && movedDown) {
                    // Drug should lower this marker and it did, but stayed normal.
                    status = 'improved';
                } else if (expectedDir === 'increase' && movedUp) {
                    // Drug should raise this marker and it did, but stayed normal.
                    status = 'improved';
                } else {
                    // Moved opposite to drug effect, but still in a safe range.
                    status = 'stable';
                    // We deliberately do NOT treat this as pathological anymore.
                }
            } else {
                // Not a drug target or change is tiny → just stable while in normal range.
                status = 'stable';
            }
        }

        if (status === 'improved') improvedCount++;
        if (status === 'worsened') worsenedCount++;

        metrics.push({
            labCode,
            name: def.name,
            unit: def.unit,
            baseline,
            finalVal,
            delta,
            deltaPercent,
            status,
            statusReason,
            values,
            normal: def.normal
        });
    }

    metrics.sort((a, b) => {
        const order = { worsened: 0, improved: 1, stable: 2 };
        if (order[a.status] !== order[b.status]) return order[a.status] - order[b.status];
        return Math.abs(b.deltaPercent) - Math.abs(a.deltaPercent);
    });

    return { metrics, improvedCount, worsenedCount };
}

function renderSingleResult() {
    const { times, series, baselines, drugConcs, patient, drug, dosage } = simulationResult;

    chartInstances.forEach(c => c.destroy());
    chartInstances = [];

    const bmi = patient.weight / Math.pow(patient.height / 100, 2);
    document.getElementById('summary-patient').textContent =
        `${patient.age}yo ${patient.sex === 'M' ? 'Male' : 'Female'}, BMI ${bmi.toFixed(1)}`;
    document.getElementById('summary-drug').textContent =
        `${drug.name}, ${dosage} mg`;

    const { metrics, improvedCount, worsenedCount } = classifyMetricsForSimulation(times, series, drug);

    document.getElementById('summary-improved').textContent = `${improvedCount} markers`;
    document.getElementById('summary-worsened').textContent = `${worsenedCount} markers`;

    document.getElementById('filter-tabs').style.display = '';

    renderMetricCards(metrics, times);
    renderCharts(metrics, times, drugConcs);
}

function renderCohortResults() {
    const {
        patients,
        perPatient,
        labStats,
        summary,
        ageInsights,
        sexInsights,
        drugInstability,
        baseDrug,
        dosage,
    } = cohortResult;

    chartInstances.forEach(c => c.destroy());
    chartInstances = [];

    const filterTabs = document.getElementById('filter-tabs');
    if (filterTabs) filterTabs.style.display = 'none';

    const cohortEl = document.getElementById('cohort-insights');
    if (cohortEl) cohortEl.style.display = 'grid';

    const numPatients = summary.patientCount;
    const numMale = patients.filter(p => p.sex === 'M').length;
    const numFemale = numPatients - numMale;
    const youngest = Math.min(...patients.map(p => p.age));
    const oldest = Math.max(...patients.map(p => p.age));

    document.getElementById('summary-patient').textContent =
        `${numPatients} synthetic patients (${numFemale} F / ${numMale} M), age ${youngest}–${oldest}`;
    document.getElementById('summary-drug').textContent =
        `${baseDrug.name}, ${dosage} mg (cohort mode)`;

    const avgPhysPerPatient = summary.totalPhysiological / numPatients;
    const avgPathPerPatient = summary.totalPathological / numPatients;
    const physRate = summary.totalPhysiological / summary.totalObservations;
    const pathRate = summary.totalPathological / summary.totalObservations;

    document.getElementById('summary-improved').textContent =
        `${avgPhysPerPatient.toFixed(1)} physiological markers / patient (${(physRate * 100).toFixed(1)}%)`;
    document.getElementById('summary-worsened').textContent =
        `${avgPathPerPatient.toFixed(1)} pathological markers / patient (${(pathRate * 100).toFixed(1)}%)`;

    const ageContainer = document.getElementById('cohort-age-insights');
    const sexContainer = document.getElementById('cohort-sex-insights');
    const featureContainer = document.getElementById('cohort-feature-instability');

    if (ageContainer) {
        ageContainer.innerHTML = '';
        ageInsights.forEach(({ group, top }) => {
            if (!top || top.rate === 0) return;
            const div = document.createElement('div');
            div.className = 'cohort-pill';
            div.innerHTML = `
                <span class="cohort-pill-label">${group.label}</span>
                <span class="cohort-pill-value">${(top.rate * 100).toFixed(1)}% pathological ${top.name}</span>
            `;
            ageContainer.appendChild(div);
        });
    }

    if (sexContainer) {
        sexContainer.innerHTML = '';
        sexInsights.forEach(({ sex, top }) => {
            if (!top || top.rate === 0) return;
            const label = sex === 'F' ? 'Women' : 'Men';
            const div = document.createElement('div');
            div.className = 'cohort-pill';
            div.innerHTML = `
                <span class="cohort-pill-label">${label}</span>
                <span class="cohort-pill-value">${(top.rate * 100).toFixed(1)}% pathological ${top.name}</span>
            `;
            sexContainer.appendChild(div);
        });
    }

    if (featureContainer) {
        featureContainer.innerHTML = '';
        const topFeatures = labStats
            .map(l => ({
                labCode: l.labCode,
                name: l.name,
                rate: l.total ? l.pathological / l.total : 0,
            }))
            .filter(f => f.rate > 0)
            .sort((a, b) => b.rate - a.rate)
            .slice(0, 3);

        topFeatures.forEach(({ name, labCode, rate }) => {
            const div = document.createElement('div');
            div.className = 'cohort-pill';
            div.innerHTML = `
                <span class="cohort-pill-label">${name} (${labCode})</span>
                <span class="cohort-pill-value">${(rate * 100).toFixed(1)}% pathological</span>
            `;
            featureContainer.appendChild(div);
        });
    }

    const grid = document.getElementById('metrics-grid');
    const chartsSection = document.getElementById('charts-section');
    if (grid) grid.innerHTML = '';
    if (chartsSection) chartsSection.innerHTML = '';

    if (chartsSection) {
        renderCohortCharts({ patients, perPatient, labStats, summary, ageInsights, sexInsights, baseDrug, dosage });
    }
}

// ============================================================
// EXPORT / DOWNLOAD
// ============================================================
async function downloadSimulationZip() {
    if (typeof JSZip === 'undefined') {
        showToast('Download library (JSZip) is not loaded. Please check your connection and reload the page.', 'error');
        return;
    }

    if (!simulationResult && !cohortResult) {
        showToast('Run a simulation first before downloading data.', 'error');
        return;
    }

    const zip = new JSZip();
    const now = new Date().toISOString();

    if (simulationMode === 'single' && simulationResult) {
        const { times, series, patient, drug, dosage, durationSec } = simulationResult;
        const { metrics } = classifyMetricsForSimulation(times, series, drug);

        const metadata = {
            simulationMode: 'single',
            generatedAt: now,
            patient,
            drug: {
                name: drug.name,
                smiles: drug.smiles,
                drugClass: drug.drugClass,
            },
            dosageMg: dosage,
            durationSec,
            metricsCount: metrics.length,
        };
        zip.file('metadata.json', JSON.stringify(metadata, null, 2));

        const esc = (value) => {
            if (value === null || value === undefined) return '';
            const str = String(value);
            if (/[",\n]/.test(str)) {
                return `"${str.replace(/"/g, '""')}"`;
            }
            return str;
        };

        const metricLines = [
            'labCode,name,unit,baseline,final,delta,deltaPercent,status',
            ...metrics.map(m =>
                [
                    esc(m.labCode),
                    esc(m.name),
                    esc(m.unit),
                    esc(m.baseline.toFixed(4)),
                    esc(m.finalVal.toFixed(4)),
                    esc(m.delta.toFixed(4)),
                    esc(m.deltaPercent.toFixed(4)),
                    esc(m.status),
                ].join(',')
            ),
        ];
        zip.file('metrics_summary.csv', metricLines.join('\n'));

        const tsLines = ['time_sec,labCode,value'];
        times.forEach((t, idx) => {
            for (const [labCode, values] of Object.entries(series)) {
                const v = values[idx];
                tsLines.push(`${t},${labCode},${v.toFixed(4)}`);
            }
        });
        zip.file('time_series.csv', tsLines.join('\n'));
    } else if (simulationMode === 'cohort' && cohortResult) {
        const {
            patients,
            perPatient,
            labStats,
            summary,
            ageInsights,
            sexInsights,
            baseDrug,
            dosage,
            durationSec,
        } = cohortResult;

        const metadata = {
            simulationMode: 'cohort',
            generatedAt: now,
            cohortSize: summary.patientCount,
            drug: {
                name: baseDrug.name,
                smiles: baseDrug.smiles,
                drugClass: baseDrug.drugClass,
            },
            dosageMg: dosage,
            durationSec,
            summary,
        };
        zip.file('metadata.json', JSON.stringify(metadata, null, 2));

        const esc = (value) => {
            if (value === null || value === undefined) return '';
            const str = String(value);
            if (/[",\n]/.test(str)) {
                return `"${str.replace(/"/g, '""')}"`;
            }
            return str;
        };

        const patientLines = [
            'id,sex,age,height_cm,weight_kg,bmi,age_group',
            ...patients.map((p, idx) =>
                [
                    idx + 1,
                    esc(p.sex),
                    esc(p.age),
                    esc(p.height.toFixed(2)),
                    esc(p.weight.toFixed(2)),
                    esc(p.bmi.toFixed(2)),
                    esc(p.ageGroup ? p.ageGroup.label : ''),
                ].join(',')
            ),
        ];
        zip.file('patients.csv', patientLines.join('\n'));

        const labLines = [
            'labCode,name,unit,total,physiological,pathological,pathologicalRate',
            ...labStats.map(l => {
                const total = l.total || 0;
                const pathological = l.pathological || 0;
                const physiological = l.physiological || 0;
                const rate = total ? pathological / total : 0;
                return [
                    esc(l.labCode),
                    esc(l.name),
                    esc(l.unit || ''),
                    esc(total),
                    esc(physiological),
                    esc(pathological),
                    esc(rate.toFixed(4)),
                ].join(',');
            }),
        ];
        zip.file('lab_stats.csv', labLines.join('\n'));

        const ageLines = [
            'ageGroup,labCode,labName,pathologicalRate',
            ...ageInsights
                .filter(a => a.top && a.top.rate > 0)
                .map(({ group, top }) =>
                    [
                        esc(group.label),
                        esc(top.labCode),
                        esc(top.name),
                        esc(top.rate.toFixed(4)),
                    ].join(',')
                ),
        ];
        zip.file('age_insights.csv', ageLines.join('\n'));

        const sexLines = [
            'sex,labCode,labName,pathologicalRate',
            ...sexInsights
                .filter(s => s.top && s.top.rate > 0)
                .map(({ sex, top }) =>
                    [
                        esc(sex === 'F' ? 'Female' : 'Male'),
                        esc(top.labCode),
                        esc(top.name),
                        esc(top.rate.toFixed(4)),
                    ].join(',')
                ),
        ];
        zip.file('sex_insights.csv', sexLines.join('\n'));

        const featureLines = [
            'labCode,labName,pathologicalRate',
            ...labStats
                .map(l => {
                    const total = l.total || 0;
                    const pathological = l.pathological || 0;
                    const rate = total ? pathological / total : 0;
                    return { labCode: l.labCode, name: l.name, rate };
                })
                .filter(f => f.rate > 0)
                .sort((a, b) => b.rate - a.rate)
                .map(f =>
                    [
                        esc(f.labCode),
                        esc(f.name),
                        esc(f.rate.toFixed(4)),
                    ].join(',')
                ),
        ];
        zip.file('most_unstable_features.csv', featureLines.join('\n'));

        // Optional per-patient time series (can be large)
        const includeTimeSeries =
            perPatient && perPatient.length
                ? await promptIncludeCohortTimeSeries()
                : false;

        if (includeTimeSeries && perPatient && perPatient.length) {
            const tsFolder = zip.folder('time_series');
            perPatient.forEach((entry, idx) => {
                const { patient, sim } = entry;
                if (!sim || !sim.times || !sim.series) return;

                const patientId = idx + 1;
                const sex = patient.sex;
                const age = patient.age;
                const height = patient.height;
                const weight = patient.weight;
                const bmi = patient.bmi;
                const ageGroup = patient.ageGroup ? patient.ageGroup.label : '';

                const lines = ['patient_id,sex,age,height_cm,weight_kg,bmi,age_group,time_sec,labCode,value'];
                sim.times.forEach((t, ti) => {
                    for (const [labCode, values] of Object.entries(sim.series)) {
                        const v = values[ti];
                        lines.push([
                            patientId,
                            sex,
                            age,
                            height.toFixed(2),
                            weight.toFixed(2),
                            bmi.toFixed(2),
                            ageGroup,
                            t,
                            labCode,
                            v.toFixed(4),
                        ].join(','));
                    }
                });
                const fileName = `patient_${patientId}_time_series.csv`;
                tsFolder.file(fileName, lines.join('\n'));
            });
        }
    } else {
        showToast('No simulation data available to export.', 'error');
        return;
    }

    const filename =
        simulationMode === 'cohort'
            ? 'dainai_cohort_simulation.zip'
            : 'dainai_single_simulation.zip';

    const blob = await zip.generateAsync({ type: 'blob' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 0);
}

function promptIncludeCohortTimeSeries() {
    return new Promise((resolve) => {
        const backdrop = document.getElementById('timeseries-modal-backdrop');
        const btnInclude = document.getElementById('timeseries-modal-include');
        const btnSkip = document.getElementById('timeseries-modal-skip');
        if (!backdrop || !btnInclude || !btnSkip) {
            resolve(false);
            return;
        }

        const cleanup = () => {
            backdrop.style.display = 'none';
            btnInclude.removeEventListener('click', onInclude);
            btnSkip.removeEventListener('click', onSkip);
        };

        const onInclude = () => {
            cleanup();
            resolve(true);
        };
        const onSkip = () => {
            cleanup();
            resolve(false);
        };

        btnInclude.addEventListener('click', onInclude);
        btnSkip.addEventListener('click', onSkip);
        backdrop.style.display = 'flex';
    });
}

function showToast(message, type = 'error', timeout = 3500) {
    const root = document.getElementById('toast-root');
    if (!root) {
        console[type === 'error' ? 'error' : 'log'](message);
        return;
    }

    const toast = document.createElement('div');
    toast.className = `toast ${type === 'error' ? 'toast-error' : ''}`;

    const badge = document.createElement('div');
    badge.className = 'toast-badge';
    badge.textContent = type === 'error' ? 'Error' : 'Info';

    const body = document.createElement('div');
    body.className = 'toast-message';
    body.textContent = message;

    toast.appendChild(badge);
    toast.appendChild(body);
    root.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(-4px)';
        setTimeout(() => {
            if (toast.parentNode === root) {
                root.removeChild(toast);
            }
        }, 200);
    }, timeout);
}
// ============================================================
// METRIC CARDS
// ============================================================
function renderMetricCards(metrics, times) {
    const grid = document.getElementById('metrics-grid');
    grid.innerHTML = '';

    metrics.forEach((m, idx) => {
        const card = document.createElement('div');
        card.className = `metric-card ${m.status} animate-in`;
        card.dataset.status = m.status;
        card.style.animationDelay = `${idx * 0.04}s`;

        const deltaSign = m.delta >= 0 ? '+' : '';
        const deltaClass = m.delta > 0.01 ? 'positive' : m.delta < -0.01 ? 'negative' : 'zero';
        const statusLabel = m.status === 'worsened' ? 'Pathological' : 'Physiological';
        const reasonHtml = m.status === 'worsened' && m.statusReason
            ? `<span class="metric-status-reason" title="${m.statusReason}">${m.statusReason}</span>`
            : '';

        card.innerHTML = `
            <div class="metric-card-header">
                <span class="metric-name">${m.name}</span>
                <div class="metric-badge-wrap">
                    <span class="metric-status-badge ${m.status}">${m.status === 'improved' ? '✓' : m.status === 'worsened' ? '✗' : '—'} ${statusLabel}</span>
                    ${reasonHtml}
                </div>
            </div>
            <div class="metric-values">
                <span class="metric-baseline">${formatValue(m.baseline, m.labCode)}</span>
                <span class="metric-delta ${deltaClass}">${deltaSign}${formatValue(m.delta, m.labCode)} ${m.unit}</span>
            </div>
            <div class="metric-code">${m.labCode}${m.normal ? ` · Normal: ${m.normal[0]}–${m.normal[1]}` : ''}</div>
            <canvas class="metric-sparkline" id="sparkline-${m.labCode}"></canvas>
        `;

        card.addEventListener('click', () => {
            const chartEl = document.getElementById(`chart-${m.labCode}`);
            if (chartEl) chartEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
        });

        grid.appendChild(card);

        // Draw sparkline
        requestAnimationFrame(() => drawSparkline(m, times));
    });
}

function formatValue(value, labCode) {
    if (['LBDSCR'].includes(labCode)) return value.toFixed(2);
    if (['LBXSK', 'LBXSCA', 'LBXSAL', 'LBXSGB', 'LBXSTP', 'LBXCRP', 'LBXBCD', 'LBXBPB', 'LBXSUA', 'LBDSCASI'].includes(labCode))
        return value.toFixed(1);
    return Math.round(value).toString();
}

function drawSparkline(metric, times) {
    const canvas = document.getElementById(`sparkline-${metric.labCode}`);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * 2;
    canvas.height = rect.height * 2;
    ctx.scale(2, 2);

    const w = rect.width;
    const h = rect.height;
    const values = metric.values;
    const min = Math.min(...values) * 0.998;
    const max = Math.max(...values) * 1.002;
    const range = max - min || 1;

    // Gradient fill
    const gradient = ctx.createLinearGradient(0, 0, 0, h);
    const color = metric.status === 'improved' ? '16,185,129' :
        metric.status === 'worsened' ? '239,68,68' : '245,158,11';
    gradient.addColorStop(0, `rgba(${color}, 0.3)`);
    gradient.addColorStop(1, `rgba(${color}, 0.02)`);

    ctx.beginPath();
    ctx.moveTo(0, h);

    for (let i = 0; i < values.length; i++) {
        const x = (i / (values.length - 1)) * w;
        const y = h - ((values[i] - min) / range) * (h - 4) - 2;
        if (i === 0) ctx.lineTo(x, y);
        else ctx.lineTo(x, y);
    }

    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Line on top
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {
        const x = (i / (values.length - 1)) * w;
        const y = h - ((values[i] - min) / range) * (h - 4) - 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = `rgba(${color}, 0.8)`;
    ctx.lineWidth = 1.5;
    ctx.stroke();
}

// ============================================================
// CHART.JS CHARTS
// ============================================================
function renderCharts(metrics, times, drugConcs) {
    const section = document.getElementById('charts-section');
    section.innerHTML = '';

    // Time labels in human-readable format
    const timeLabels = times.map(t => formatTime(t));

    // First: Drug Concentration chart
    renderDrugConcChart(section, times, timeLabels, drugConcs);

    // Then: each metric chart
    metrics.forEach(metric => {
        const card = document.createElement('div');
        card.className = 'chart-card';
        card.id = `chart-${metric.labCode}`;

        const statusEmoji = metric.status === 'improved' ? '✓' : metric.status === 'worsened' ? '✗' : '—';
        const statusLabel = metric.status === 'worsened' ? 'Pathological' : 'Physiological';
        const reasonHtml = metric.status === 'worsened' && metric.statusReason
            ? `<span class="metric-status-reason" title="${metric.statusReason}">${metric.statusReason}</span>`
            : '';

        card.innerHTML = `
            <div class="chart-card-header">
                <div>
                    <div class="chart-title">${metric.name}</div>
                    <div class="chart-subtitle">${metric.labCode} · ${metric.unit}${metric.normal ? ` · Normal: ${metric.normal[0]}–${metric.normal[1]}` : ''}</div>
                </div>
                <div class="metric-badge-wrap">
                    <span class="metric-status-badge ${metric.status}">${statusEmoji} ${statusLabel}</span>
                    ${reasonHtml}
                </div>
            </div>
            <div class="chart-container">
                <canvas id="canvas-${metric.labCode}"></canvas>
            </div>
        `;

        section.appendChild(card);

        requestAnimationFrame(() => {
            createMetricChart(metric, timeLabels);
        });
    });
}

// ============================================================
// COHORT CHARTS (PATIENT COUNT > 1)
// ============================================================
function renderCohortCharts({ patients, perPatient, labStats, summary, ageInsights, sexInsights, baseDrug, dosage }) {
    const section = document.getElementById('charts-section');
    if (!section) return;

    section.innerHTML = '';

    const primaryLabCode = getPrimaryLabCodeForDrug(baseDrug, labStats);

    if (primaryLabCode && perPatient && perPatient.length) {
        renderCohortTimeDynamics(section, perPatient, primaryLabCode);
        renderCohortDistributionShift(section, perPatient, primaryLabCode);
    }

    renderCohortPathologyByGroup(section, labStats);
    renderCohortInstabilityHeatmap(section, perPatient, labStats);
    renderCohortFeatureRiskRanking(section, labStats);
}

function getPrimaryLabCodeForDrug(drug, labStats) {
    if (drug && drug.primaryEffect && LAB_DEFINITIONS[drug.primaryEffect.lab]) {
        return drug.primaryEffect.lab;
    }
    if (labStats && labStats.length) {
        const sorted = [...labStats].sort((a, b) => {
            const ra = a.total ? a.pathological / a.total : 0;
            const rb = b.total ? b.pathological / b.total : 0;
            return rb - ra;
        });
        return sorted[0]?.labCode || null;
    }
    return null;
}

function buildMeanTimeSeries(perPatient, labCode) {
    if (!perPatient || !perPatient.length) return null;
    const firstSim = perPatient[0].sim;
    if (!firstSim || !firstSim.times || !firstSim.series[labCode]) return null;

    const times = firstSim.times;
    const meanValues = times.map((_, idx) => {
        let sum = 0;
        let count = 0;
        perPatient.forEach(({ sim }) => {
            if (!sim || !sim.series[labCode]) return;
            const arr = sim.series[labCode];
            if (idx < arr.length) {
                sum += arr[idx];
                count += 1;
            }
        });
        return count ? sum / count : null;
    });

    return { times, meanValues };
}

function renderCohortTimeDynamics(section, perPatient, labCode) {
    const def = LAB_DEFINITIONS[labCode];
    const agg = buildMeanTimeSeries(perPatient, labCode);
    if (!def || !agg) return;

    const timeLabels = agg.times.map(t => formatTime(t));

    const card = document.createElement('div');
    card.className = 'chart-card';
    card.innerHTML = `
        <div class="chart-card-header">
            <div>
                <div class="chart-title">Cohort Time Dynamics — ${def.name}</div>
                <div class="chart-subtitle">
                    ${labCode} · Mean value over time after dose
                    ${def.unit ? ` · Units: ${def.unit}` : ''}
                    ${def.normal ? ` · Normal band: ${def.normal[0]}–${def.normal[1]} ${def.unit || ''}` : ''}
                </div>
            </div>
        </div>
        <div class="chart-container">
            <canvas id="cohort-time-dynamics"></canvas>
        </div>
    `;
    section.appendChild(card);

    requestAnimationFrame(() => {
        const ctx = document.getElementById('cohort-time-dynamics');
        if (!ctx) return;
        const context = ctx.getContext('2d');

        const gradient = context.createLinearGradient(0, 0, 0, 300);
        gradient.addColorStop(0, hexToRgba('#22c55e', 0.3));
        gradient.addColorStop(1, hexToRgba('#22c55e', 0.02));

        const datasets = [{
            label: def.name,
            data: agg.meanValues,
            borderColor: '#22c55e',
            backgroundColor: gradient,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
            fill: true,
        }];

        if (def.normal) {
            const low = agg.meanValues.map(() => def.normal[0]);
            const high = agg.meanValues.map(() => def.normal[1]);
            datasets.push({
                label: 'Normal range (low)',
                data: low,
                borderColor: 'rgba(148, 163, 184, 0)',
                backgroundColor: hexToRgba('#0ea5e9', 0.06),
                pointRadius: 0,
                borderWidth: 0,
                fill: '+1',
            });
            datasets.push({
                label: 'Normal range (high)',
                data: high,
                borderColor: 'rgba(148, 163, 184, 0)',
                backgroundColor: hexToRgba('#0ea5e9', 0.06),
                pointRadius: 0,
                borderWidth: 0,
                fill: false,
            });
        }

        const baseOptions = chartOptions(def.unit || '');
        baseOptions.scales.x.title = {
            display: true,
            text: 'Time after dose',
            color: '#94a3b8',
            font: { family: 'Inter', size: 11 },
        };
        baseOptions.scales.y.title = {
            display: true,
            text: `Mean ${def.name}${def.unit ? ` (${def.unit})` : ''}`,
            color: '#94a3b8',
            font: { family: 'Inter', size: 11 },
        };

        const chart = new Chart(context, {
            type: 'line',
            data: { labels: timeLabels, datasets },
            options: baseOptions,
        });
        chartInstances.push(chart);
    });
}

function renderCohortDistributionShift(section, perPatient, labCode) {
    const def = LAB_DEFINITIONS[labCode];
    if (!def || !perPatient || !perPatient.length) return;

    const baselines = [];
    const finals = [];
    perPatient.forEach(({ sim }) => {
        if (!sim || !sim.series[labCode]) return;
        const arr = sim.series[labCode];
        if (!arr.length) return;
        baselines.push(arr[0]);
        finals.push(arr[arr.length - 1]);
    });
    if (!baselines.length || !finals.length) return;

    const all = baselines.concat(finals);
    const minVal = Math.min(...all);
    const maxVal = Math.max(...all);
    const binCount = 20;
    const binSize = (maxVal - minVal) / binCount || 1;

    const baselineBins = new Array(binCount).fill(0);
    const finalBins = new Array(binCount).fill(0);

    const assignBin = (value) => {
        let idx = Math.floor((value - minVal) / binSize);
        if (idx < 0) idx = 0;
        if (idx >= binCount) idx = binCount - 1;
        return idx;
    };

    baselines.forEach(v => {
        baselineBins[assignBin(v)] += 1;
    });
    finals.forEach(v => {
        finalBins[assignBin(v)] += 1;
    });

    const labels = [];
    for (let i = 0; i < binCount; i++) {
        const start = minVal + i * binSize;
        const end = start + binSize;
        labels.push(`${start.toFixed(1)}–${end.toFixed(1)}`);
    }

    const card = document.createElement('div');
    card.className = 'chart-card';
    card.innerHTML = `
        <div class="chart-card-header">
            <div>
                <div class="chart-title">Distribution Shift — ${def.name}</div>
                <div class="chart-subtitle">
                    ${labCode} · X-axis: value bins (${def.unit || 'units'}) · Y-axis: number of patients
                </div>
            </div>
        </div>
        <div class="chart-container">
            <canvas id="cohort-distribution-shift"></canvas>
        </div>
    `;
    section.appendChild(card);

    requestAnimationFrame(() => {
        const ctx = document.getElementById('cohort-distribution-shift');
        if (!ctx) return;
        const chart = new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels,
                datasets: [
                    {
                        label: 'Baseline',
                        data: baselineBins,
                        backgroundColor: hexToRgba('#38bdf8', 0.6),
                        borderColor: '#38bdf8',
                        borderWidth: 1,
                    },
                    {
                        label: 'Final',
                        data: finalBins,
                        backgroundColor: hexToRgba('#f97316', 0.6),
                        borderColor: '#f97316',
                        borderWidth: 1,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: '#e2e8f0', font: { family: 'Inter', size: 11 } },
                    },
                },
                scales: {
                    x: {
                        stacked: false,
                        ticks: { color: '#64748b', maxRotation: 45, minRotation: 45, font: { family: 'Inter', size: 9 } },
                        grid: { display: false },
                        title: {
                            display: true,
                            text: `${def.name} (${def.unit || 'units'})`,
                            color: '#94a3b8',
                            font: { family: 'Inter', size: 11 },
                        },
                    },
                    y: {
                        ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 11 } },
                        grid: { color: 'rgba(148,163,184,0.08)' },
                        title: {
                            display: true,
                            text: 'Number of patients',
                            color: '#94a3b8',
                            font: { family: 'Inter', size: 11 },
                        },
                    },
                },
            },
        });
        chartInstances.push(chart);
    });
}

function renderCohortPathologyByGroup(section, labStats) {
    if (!labStats || !labStats.length) return;

    const ageRates = AGE_GROUPS.map(group => {
        let total = 0;
        let pathological = 0;
        labStats.forEach(l => {
            const bucket = l.byAgeGroup[group.id];
            if (!bucket) return;
            total += bucket.total || 0;
            pathological += bucket.pathological || 0;
        });
        const rate = total ? pathological / total : 0;
        return { label: group.label, rate };
    });

    const sexes = ['F', 'M'];
    const sexRates = sexes.map(sex => {
        let total = 0;
        let pathological = 0;
        labStats.forEach(l => {
            const bucket = l.bySex[sex];
            if (!bucket) return;
            total += bucket.total || 0;
            pathological += bucket.pathological || 0;
        });
        const rate = total ? pathological / total : 0;
        return { label: sex === 'F' ? 'Women' : 'Men', rate };
    });

    const card = document.createElement('div');
    card.className = 'chart-card';
    card.innerHTML = `
        <div class="chart-card-header">
            <div>
                <div class="chart-title">Pathological Rate by Cohort</div>
                <div class="chart-subtitle">Fraction of all lab markers that end pathological in each subgroup (0–100%)</div>
            </div>
        </div>
        <div class="chart-container" style="height: 260px; display: grid; grid-template-columns: 1fr 1fr; gap: 24px;">
            <canvas id="cohort-age-bars"></canvas>
            <canvas id="cohort-sex-bars"></canvas>
        </div>
    `;
    section.appendChild(card);

    requestAnimationFrame(() => {
        const ageCtx = document.getElementById('cohort-age-bars');
        const sexCtx = document.getElementById('cohort-sex-bars');
        if (ageCtx) {
            const chart = new Chart(ageCtx.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: ageRates.map(a => a.label),
                    datasets: [{
                        label: '% pathological markers',
                        data: ageRates.map(a => a.rate * 100),
                        backgroundColor: hexToRgba('#f97316', 0.7),
                    }],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: {
                            ticks: { color: '#64748b', font: { family: 'Inter', size: 10 } },
                            grid: { display: false },
                            title: {
                                display: true,
                                text: 'Age group (years)',
                                color: '#94a3b8',
                                font: { family: 'Inter', size: 11 },
                            },
                        },
                        y: {
                            ticks: {
                                color: '#64748b',
                                font: { family: 'JetBrains Mono', size: 10 },
                                callback: v => `${v}%`,
                            },
                            grid: { color: 'rgba(148,163,184,0.08)' },
                            suggestedMin: 0,
                            suggestedMax: 100,
                            title: {
                                display: true,
                                text: '% pathological markers',
                                color: '#94a3b8',
                                font: { family: 'Inter', size: 11 },
                            },
                        },
                    },
                },
            });
            chartInstances.push(chart);
        }
        if (sexCtx) {
            const chart = new Chart(sexCtx.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: sexRates.map(s => s.label),
                    datasets: [{
                        label: '% pathological markers',
                        data: sexRates.map(s => s.rate * 100),
                        backgroundColor: hexToRgba('#6366f1', 0.8),
                    }],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: {
                            ticks: { color: '#64748b', font: { family: 'Inter', size: 10 } },
                            grid: { display: false },
                            title: {
                                display: true,
                                text: 'Sex',
                                color: '#94a3b8',
                                font: { family: 'Inter', size: 11 },
                            },
                        },
                        y: {
                            ticks: {
                                color: '#64748b',
                                font: { family: 'JetBrains Mono', size: 10 },
                                callback: v => `${v}%`,
                            },
                            grid: { color: 'rgba(148,163,184,0.08)' },
                            suggestedMin: 0,
                            suggestedMax: 100,
                            title: {
                                display: true,
                                text: '% pathological markers',
                                color: '#94a3b8',
                                font: { family: 'Inter', size: 11 },
                            },
                        },
                    },
                },
            });
            chartInstances.push(chart);
        }
    });
}

function renderCohortInstabilityHeatmap(section, perPatient, labStats) {
    if (!perPatient || !perPatient.length || !labStats || !labStats.length) return;

    const analysisSize = Math.min(perPatient.length, 200);
    const subset = perPatient.slice(0, analysisSize);
    const firstSim = subset[0].sim;
    if (!firstSim || !firstSim.times) return;
    const times = firstSim.times;

    const rankedLabs = [...labStats]
        .map(l => {
            const rate = l.total ? l.pathological / l.total : 0;
            return { labCode: l.labCode, name: l.name, rate };
        })
        .filter(l => l.rate > 0)
        .sort((a, b) => b.rate - a.rate);

    const topLabs = rankedLabs.slice(0, 8);
    if (!topLabs.length) return;

    const rows = topLabs.map(l => {
        const def = LAB_DEFINITIONS[l.labCode] || {};
        const normal = def.normal;
        const rates = times.map((t, ti) => {
            let total = 0;
            let pathological = 0;
            subset.forEach(({ sim }) => {
                if (!sim || !sim.series[l.labCode]) return;
                const arr = sim.series[l.labCode];
                if (ti >= arr.length) return;
                const v = arr[ti];
                total += 1;
                if (normal && (v < normal[0] || v > normal[1])) {
                    pathological += 1;
                }
            });
            return total ? pathological / total : 0;
        });
        return { labCode: l.labCode, name: def.name || l.name, rates };
    });

    const card = document.createElement('div');
    card.className = 'chart-card';
    card.innerHTML = `
        <div class="chart-card-header">
            <div>
                <div class="chart-title">Instability Overview — Heatmap</div>
                <div class="chart-subtitle">Biomarker × Time · Color = % pathological (subset of cohort)</div>
            </div>
        </div>
        <div class="heatmap-container">
            <div class="heatmap-legend">
                <span>Stable</span>
                <div class="heatmap-legend-bar"></div>
                <span>Unstable</span>
            </div>
            <div class="heatmap-rows" id="cohort-heatmap-rows"></div>
        </div>
    `;
    section.appendChild(card);

    const rowsRoot = card.querySelector('#cohort-heatmap-rows');
    if (!rowsRoot) return;

    const maxCells = 40;
    const step = Math.max(1, Math.floor(times.length / maxCells));

    rows.forEach(row => {
        const rowEl = document.createElement('div');
        rowEl.className = 'heatmap-row';
        const label = document.createElement('div');
        label.className = 'heatmap-label';
        label.textContent = row.name || row.labCode;
        const cells = document.createElement('div');
        cells.className = 'heatmap-row-cells';

        row.rates.forEach((rate, idx) => {
            if (idx % step !== 0) return;
            const cell = document.createElement('div');
            cell.className = 'heatmap-cell';
            const intensity = rate;
            const alpha = 0.1 + 0.8 * intensity;
            cell.style.backgroundColor = `rgba(239,68,68,${alpha})`;
            cells.appendChild(cell);
        });

        rowEl.appendChild(label);
        rowEl.appendChild(cells);
        rowsRoot.appendChild(rowEl);
    });
}

function renderCohortFeatureRiskRanking(section, labStats) {
    if (!labStats || !labStats.length) return;
    const ranked = [...labStats]
        .map(l => {
            const rate = l.total ? l.pathological / l.total : 0;
            return { labCode: l.labCode, name: l.name, rate };
        })
        .filter(l => l.rate > 0)
        .sort((a, b) => b.rate - a.rate)
        .slice(0, 10);

    if (!ranked.length) return;

    const labels = ranked.map(l => `${l.name} (${l.labCode})`);
    const data = ranked.map(l => l.rate * 100);

    const card = document.createElement('div');
    card.className = 'chart-card';
    card.innerHTML = `
        <div class="chart-card-header">
            <div>
                <div class="chart-title">Feature Risk Ranking</div>
                <div class="chart-subtitle">Biomarkers ranked by instability (% pathological)</div>
            </div>
        </div>
        <div class="chart-container">
            <canvas id="cohort-feature-ranking"></canvas>
        </div>
    `;
    section.appendChild(card);

    requestAnimationFrame(() => {
        const ctx = document.getElementById('cohort-feature-ranking');
        if (!ctx) return;
        const chart = new Chart(ctx.getContext('2d'), {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: '% pathological',
                    data,
                    backgroundColor: hexToRgba('#ef4444', 0.8),
                }],
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: {
                        ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 11 } },
                        grid: { color: 'rgba(148,163,184,0.08)' },
                    },
                    y: {
                        ticks: { color: '#cbd5f5', font: { family: 'Inter', size: 10 } },
                        grid: { display: false },
                    },
                },
            },
        });
        chartInstances.push(chart);
    });
}

function renderDrugConcChart(section, times, timeLabels, drugConcs) {
    const card = document.createElement('div');
    card.className = 'chart-card';
    card.innerHTML = `
        <div class="chart-card-header">
            <div>
                <div class="chart-title">Drug Concentration (PK Profile)</div>
                <div class="chart-subtitle">mg/L · One-compartment IV bolus model</div>
            </div>
        </div>
        <div class="chart-container">
            <canvas id="canvas-drug-conc"></canvas>
        </div>
    `;
    section.appendChild(card);

    requestAnimationFrame(() => {
        const ctx = document.getElementById('canvas-drug-conc').getContext('2d');
        const gradient = ctx.createLinearGradient(0, 0, 0, 300);
        gradient.addColorStop(0, 'rgba(124, 77, 255, 0.35)');
        gradient.addColorStop(1, 'rgba(124, 77, 255, 0.02)');

        const chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: timeLabels,
                datasets: [{
                    label: 'Drug Concentration (mg/L)',
                    data: drugConcs,
                    borderColor: '#7C4DFF',
                    backgroundColor: gradient,
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.3,
                }]
            },
            options: chartOptions('mg/L'),
        });
        chartInstances.push(chart);
    });
}

function createMetricChart(metric, timeLabels) {
    const ctx = document.getElementById(`canvas-${metric.labCode}`);
    if (!ctx) return;

    const context = ctx.getContext('2d');
    const color = metric.status === 'improved' ? '#10b981' :
        metric.status === 'worsened' ? '#ef4444' : '#f59e0b';

    const gradient = context.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, hexToRgba(color, 0.3));
    gradient.addColorStop(1, hexToRgba(color, 0.02));

    const datasets = [{
        label: metric.name,
        data: metric.values,
        borderColor: color,
        backgroundColor: gradient,
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 4,
        fill: true,
        tension: 0.3,
    }];

    // Add normal range bands if available
    if (metric.normal) {
        const len = metric.values.length;
        datasets.push({
            label: 'Normal High',
            data: Array(len).fill(metric.normal[1]),
            borderColor: 'rgba(148, 163, 184, 0.3)',
            borderWidth: 1,
            borderDash: [5, 5],
            pointRadius: 0,
            fill: false,
        });
        datasets.push({
            label: 'Normal Low',
            data: Array(len).fill(metric.normal[0]),
            borderColor: 'rgba(148, 163, 184, 0.3)',
            borderWidth: 1,
            borderDash: [5, 5],
            pointRadius: 0,
            fill: false,
        });
    }

    const chart = new Chart(context, {
        type: 'line',
        data: { labels: timeLabels, datasets },
        options: chartOptions(metric.unit),
    });
    chartInstances.push(chart);
}

function chartOptions(unit) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: 'rgba(17, 24, 39, 0.95)',
                titleColor: '#f0f4f8',
                bodyColor: '#94a3b8',
                borderColor: 'rgba(148, 163, 184, 0.2)',
                borderWidth: 1,
                cornerRadius: 8,
                padding: 12,
                titleFont: { family: 'Inter', size: 13, weight: 600 },
                bodyFont: { family: 'JetBrains Mono', size: 12 },
                callbacks: {
                    label: function (context) {
                        return `${context.dataset.label}: ${context.parsed.y.toFixed(2)} ${unit}`;
                    }
                }
            },
        },
        scales: {
            x: {
                grid: { color: 'rgba(148, 163, 184, 0.06)' },
                ticks: {
                    color: '#64748b',
                    font: { family: 'Inter', size: 11 },
                    maxTicksLimit: 10,
                    maxRotation: 0,
                },
            },
            y: {
                grid: { color: 'rgba(148, 163, 184, 0.06)' },
                ticks: {
                    color: '#64748b',
                    font: { family: 'JetBrains Mono', size: 11 },
                },
            },
        },
        animation: {
            duration: 800,
            easing: 'easeInOutQuart',
        },
    };
}

function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function formatTime(seconds) {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    return m > 0 ? `${h}h${m}m` : `${h}h`;
}

// ============================================================
// FILTER TABS
// ============================================================
function initFilterTabs() {
    document.getElementById('filter-tabs').addEventListener('click', (e) => {
        const tab = e.target.closest('.filter-tab');
        if (!tab) return;

        document.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        const filter = tab.dataset.filter;
        document.querySelectorAll('.metric-card').forEach(card => {
            if (filter === 'all' || card.dataset.status === filter) {
                card.style.display = '';
            } else {
                card.style.display = 'none';
            }
        });
    });
}

// ============================================================
// PARTICLES BACKGROUND
// ============================================================
function initParticles() {
    const canvas = document.getElementById('particles-canvas');
    const ctx = canvas.getContext('2d');
    let particles = [];
    const maxParticles = 60;

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    resize();
    window.addEventListener('resize', resize);

    class Particle {
        constructor() {
            this.reset();
        }
        reset() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.vx = (Math.random() - 0.5) * 0.3;
            this.vy = (Math.random() - 0.5) * 0.3;
            this.radius = Math.random() * 1.5 + 0.5;
            this.opacity = Math.random() * 0.3 + 0.1;
            this.hue = Math.random() > 0.5 ? 187 : 262; // cyan or violet
        }
        update() {
            this.x += this.vx;
            this.y += this.vy;
            if (this.x < 0 || this.x > canvas.width || this.y < 0 || this.y > canvas.height) {
                this.reset();
            }
        }
        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${this.hue}, 100%, 60%, ${this.opacity})`;
            ctx.fill();
        }
    }

    for (let i = 0; i < maxParticles; i++) {
        particles.push(new Particle());
    }

    function connectParticles() {
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 150) {
                    const opacity = (1 - dist / 150) * 0.08;
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(0, 229, 255, ${opacity})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {
            p.update();
            p.draw();
        });
        connectParticles();
        requestAnimationFrame(animate);
    }

    animate();
}
