"""
CLINICAL LAB TEST REFERENCE RANGES - ALL HUMANS

General population ranges (all ages, both sexes)

Format:
- Normal range: What 95% of healthy people have
- Clinical range: What's seen in medical practice (including disease states)
- Physiological limits: Extreme values compatible with life
"""

LAB_RANGES = {
    # ===== LIVER FUNCTION TESTS =====
    "LBXSGT": {
        "name": "Gamma Glutamyl Transferase (GGT)",
        "unit": "U/L",
        "normal_range": (5, 55),
        "clinical_range": (1, 500),
        "physiological_limit": (0, 2000),
        "sex_difference": "Males 2x higher than females",
        "notes": "Alcohol/drugs increase dramatically. NEVER negative. Severe liver disease can reach 1000+."
    },
    
    "LBXSAL": {
        "name": "Albumin",
        "unit": "g/dL",
        "normal_range": (3.4, 5.4),
        "clinical_range": (1.5, 6.0),
        "physiological_limit": (1.0, 6.5),
        "sex_difference": "None significant",
        "notes": "Main protein in blood. <2.5 = severe malnutrition/liver failure. >5.5 = dehydration."
    },
    
    "LBXSAS": {
        "name": "Aspartate Aminotransferase (AST)",
        "unit": "U/L",
        "normal_range": (5, 40),
        "clinical_range": (1, 1000),
        "physiological_limit": (0, 10000),
        "sex_difference": "Males slightly higher",
        "notes": "NEVER negative. Heart attack/liver damage can reach 1000s. Extreme: 10,000+."
    },
    
    "LBXSGB": {
        "name": "Total Bilirubin",
        "unit": "mg/dL",
        "normal_range": (0.1, 1.2),
        "clinical_range": (0.05, 25.0),
        "physiological_limit": (0.01, 50),
        "sex_difference": "Males slightly higher",
        "notes": ">3 = visible jaundice. Newborns can be 15-20. Severe liver failure: 30-50."
    },
    
    "LBXSTP": {
        "name": "Total Protein",
        "unit": "g/dL",
        "normal_range": (6.0, 8.3),
        "clinical_range": (4.0, 10.0),
        "physiological_limit": (3.0, 12.0),
        "sex_difference": "None significant",
        "notes": "<5 = severe malnutrition. >9 = dehydration/multiple myeloma."
    },
    
    # ===== KIDNEY FUNCTION TESTS =====
    "LBDSCR": {
        "name": "Creatinine (SI)",
        "unit": "μmol/L",
        "normal_range": (53, 115),
        "clinical_range": (27, 1330),
        "physiological_limit": (18, 2650),
        "sex_difference": "Males: 62-115, Females: 53-97",
        "notes": "SI version of creatinine. Divide by 88.4 for mg/dL. Dialysis patients: 440-1330. Acute kidney failure: 880-2650."
    },
    
    "LBXSBU": {
        "name": "Blood Urea Nitrogen (BUN)",
        "unit": "mg/dL",
        "normal_range": (7, 20),
        "clinical_range": (2, 100),
        "physiological_limit": (1, 200),
        "sex_difference": "Males slightly higher",
        "notes": "Dehydration: 30-50. Severe kidney failure: 80-150. Extreme: 200+."
    },
    
    "LBXSUA": {
        "name": "Uric Acid",
        "unit": "mg/dL",
        "normal_range": (2.5, 7.5),
        "clinical_range": (1.0, 13.0),
        "physiological_limit": (0.5, 20),
        "sex_difference": "Males: 3.5-7.2, Females: 2.6-6.0",
        "notes": "Gout attacks >8. Tumor lysis syndrome can reach 15-20."
    },
    
    # ===== ELECTROLYTES =====
    "LBXSNA": {
        "name": "Sodium",
        "unit": "mmol/L",
        "normal_range": (136, 145),
        "clinical_range": (120, 160),
        "physiological_limit": (110, 170),
        "sex_difference": "None",
        "notes": "CRITICAL: <120 or >160 = seizure risk. <110 or >170 = life-threatening."
    },
    
    "LBXSK": {
        "name": "Potassium",
        "unit": "mmol/L",
        "normal_range": (3.5, 5.0),
        "clinical_range": (2.5, 6.5),
        "physiological_limit": (2.0, 8.0),
        "sex_difference": "None",
        "notes": "CRITICAL: <2.5 or >6.5 = cardiac arrest risk. Most tightly regulated electrolyte."
    },
    
    "LBXSCL": {
        "name": "Chloride",
        "unit": "mmol/L",
        "normal_range": (96, 106),
        "clinical_range": (85, 115),
        "physiological_limit": (70, 130),
        "sex_difference": "None",
        "notes": "Follows sodium. Vomiting/diarrhea cause large shifts."
    },
    
    "LBXSCA": {
        "name": "Calcium",
        "unit": "mg/dL",
        "normal_range": (8.5, 10.5),
        "clinical_range": (6.0, 14.0),
        "physiological_limit": (5.0, 18),
        "sex_difference": "None significant",
        "notes": "CRITICAL: <6.5 = tetany/seizures. >13 = cardiac issues. Tightly regulated."
    },
    
    "LBXSOS": {
        "name": "Osmolality",
        "unit": "mmol/Kg",
        "normal_range": (275, 295),
        "clinical_range": (250, 320),
        "physiological_limit": (240, 350),
        "sex_difference": "None",
        "notes": "Related to sodium/hydration. Extreme dehydration: 320-350."
    },
    
    # ===== LIPID PANEL =====
    "LBXTC": {
        "name": "Total Cholesterol",
        "unit": "mg/dL",
        "normal_range": (125, 200),
        "clinical_range": (80, 400),
        "physiological_limit": (60, 600),
        "sex_difference": "Women slightly higher after menopause",
        "notes": "Desirable <200. High 200-239. Very high >240. Familial hypercholesterolemia: 400-600."
    },
    
    "LBDSCH": {
        "name": "Serum Cholesterol (SI)",
        "unit": "mmol/L",
        "normal_range": (3.6, 5.2),
        "clinical_range": (2.1, 10.3),
        "physiological_limit": (1.5, 15.5),
        "sex_difference": "Women slightly higher after menopause",
        "notes": "SI version of LBXSCH. Desirable <5.2. High 5.2-6.2. Very high >6.2. Multiply by 38.67 for mg/dL."
    },
    
    "LBDSCL": {
        "name": "LDL Cholesterol (Bad)",
        "unit": "mg/dL",
        "normal_range": (50, 130),
        "clinical_range": (20, 250),
        "physiological_limit": (10, 400),
        "sex_difference": "Men higher in middle age",
        "notes": "Optimal <100. High 160-189. Very high >190. Familial: 250-400."
    },
    
    "LBXSCH": {
        "name": "Serum Cholesterol (duplicate of TC)",
        "unit": "mg/dL",
        "normal_range": (125, 200),
        "clinical_range": (80, 400),
        "physiological_limit": (60, 600),
        "sex_difference": "Same as total cholesterol",
        "notes": "Usually same value as LBXTC - may be redundant in dataset."
    },
    
    "LBDTC": {
        "name": "Total Cholesterol (SI)",
        "unit": "mmol/L",
        "normal_range": (3.6, 5.2),
        "clinical_range": (2.1, 10.3),
        "physiological_limit": (1.5, 15.5),
        "sex_difference": "Women slightly higher after menopause",
        "notes": "SI version of LBXTC. Desirable <5.2. High 5.2-6.2. Multiply by 38.67 for mg/dL."
    },
    
    # ===== GLUCOSE & METABOLIC =====
    "LBXSGL": {
        "name": "Fasting Glucose",
        "unit": "mg/dL",
        "normal_range": (70, 100),
        "clinical_range": (40, 400),
        "physiological_limit": (20, 600),
        "sex_difference": "None significant",
        "notes": "Prediabetes: 100-125. Diabetes: ≥126. Hypoglycemia: <70. Coma risk: <40 or >600."
    },
    
    # ===== INFLAMMATION =====
    "LBXCRP": {
        "name": "C-Reactive Protein",
        "unit": "mg/L",
        "normal_range": (0.0, 3.0),
        "clinical_range": (0.0, 100),
        "physiological_limit": (0.0, 500),
        "sex_difference": "None",
        "notes": "<1 = low risk. 1-3 = moderate. >3 = high. Severe infection: 50-200. Extreme: 300-500."
    },
    
    # ===== HEAVY METALS / TOXINS =====
    "LBXBCD": {
        "name": "Blood Cadmium",
        "unit": "µg/L",
        "normal_range": (0.1, 1.5),
        "clinical_range": (0.0, 10.0),
        "physiological_limit": (0.0, 50),
        "sex_difference": "Smokers 2-3x higher",
        "notes": "Non-smokers: <0.5. Smokers: 1-3. Occupational exposure: 5-10. Acute toxicity: >20."
    },
    
    "LBXBPB": {
        "name": "Blood Lead",
        "unit": "µg/dL",
        "normal_range": (0.5, 5.0),
        "clinical_range": (0.0, 60),
        "physiological_limit": (0.0, 150),
        "sex_difference": "None",
        "notes": "No safe level. <5 acceptable. 10-20 = concern. >60 = chelation therapy. Acute: 100-150."
    },
    
    # ===== UNCLEAR ABBREVIATIONS =====
    "LBDSCASI": {
        "name": "UNKNOWN - Not standard LOINC/NHANES code",
        "unit": "Unknown",
        "normal_range": None,
        "clinical_range": None,
        "physiological_limit": None,
        "sex_difference": "Unknown",
        "notes": "May be calculated score, composite index, or typo. Not found in standard lab panels."
    },
}

# ===== CRITICAL PHYSIOLOGICAL CONSTRAINTS =====
CRITICAL_RULES = {
    "MUST_BE_POSITIVE": [
        "LBXSGT", "LBXSAS", "LBXSAL", "LBXSTP", "LBXSGB",
        "LBXSGL", "LBXCRP", "LBXTC", "LBDSCH", "LBXSCH", "LBDTC",
        "LBXSCA", "LBXBCD", "LBXBPB"
    ],
    "TIGHTLY_REGULATED": {
        "LBXSK": (2.0, 8.0),  # Potassium - cardiac arrest risk outside
        "LBXSNA": (110, 170),  # Sodium - seizures/death outside
        "LBXSCA": (5.0, 18),   # Calcium - arrhythmia outside
        "LBXSGL": (20, 600),   # Glucose - coma outside
    },
    "IMPOSSIBLE_COMBINATIONS": {
        "LDL_GT_TC": "Cannot have LDL > Total Cholesterol",
        "HDL_GT_TC": "Cannot have HDL > Total Cholesterol",
        "TC_APPROX": "Total Cholesterol ≈ LDL + HDL + (Triglycerides/5)",
        "BUN_CR_RATIO": "BUN/Creatinine ratio typically 10:1 to 20:1"
    }
}


def get_lab_range(metric_code: str, range_type: str = "normal_range", sex: str = None) -> tuple:
    """
    Get lab reference range for a metric.
    
    Args:
        metric_code: Lab metric code (e.g., 'LBXTC', 'LBXSGL')
        range_type: 'normal_range', 'clinical_range', or 'physiological_limit'
        sex: 'M' or 'F' for sex-specific adjustments
    
    Returns:
        Tuple of (min, max) values, or None if not found
    """
    if metric_code not in LAB_RANGES:
        return None
    
    info = LAB_RANGES[metric_code]
    if info[range_type] is None:
        return None
    
    min_val, max_val = info[range_type]
    
    # Apply sex-specific adjustments if available
    if sex and 'sex_difference' in info:
        # For now, return base range (can be enhanced with specific sex logic)
        pass
    
    return (min_val, max_val)


def validate_lab_value(metric_code: str, value: float, allow_clinical: bool = True) -> tuple[bool, str]:
    """
    Validate if a lab value is physiologically plausible.
    
    Args:
        metric_code: Lab metric code
        value: Value to validate
        allow_clinical: If True, allow clinical_range; if False, only normal_range
    
    Returns:
        (is_valid, message)
    """
    if metric_code not in LAB_RANGES:
        return (True, "Unknown metric - cannot validate")
    
    info = LAB_RANGES[metric_code]
    
    # Check if must be positive
    if metric_code in CRITICAL_RULES["MUST_BE_POSITIVE"]:
        if value < 0:
            return (False, f"{info['name']} must be positive (got {value})")
    
    # Check physiological limits
    if info['physiological_limit']:
        min_phys, max_phys = info['physiological_limit']
        if value < min_phys or value > max_phys:
            return (False, f"{info['name']} outside physiological limits ({min_phys}-{max_phys})")
    
    # Check if in acceptable range
    if allow_clinical and info['clinical_range']:
        min_clin, max_clin = info['clinical_range']
        if min_clin <= value <= max_clin:
            return (True, "Within clinical range")
    
    if info['normal_range']:
        min_norm, max_norm = info['normal_range']
        if min_norm <= value <= max_norm:
            return (True, "Within normal range")
    
    return (True, "Outside normal but within physiological limits")


def clamp_to_physiological_limits(metric_code: str, value: float) -> float:
    """
    Clamp a value to physiological limits.
    
    Args:
        metric_code: Lab metric code
        value: Value to clamp
    
    Returns:
        Clamped value
    """
    if metric_code not in LAB_RANGES:
        return value
    
    info = LAB_RANGES[metric_code]
    if info['physiological_limit']:
        min_phys, max_phys = info['physiological_limit']
        return max(min_phys, min(value, max_phys))
    
    return value


def get_sex_adjusted_range(metric_code: str, sex: str, range_type: str = "normal_range") -> tuple:
    """
    Get sex-adjusted reference range.
    
    Args:
        metric_code: Lab metric code
        sex: 'M' or 'F'
        range_type: 'normal_range', 'clinical_range', or 'physiological_limit'
    
    Returns:
        Tuple of (min, max) values adjusted for sex
    """
    base_range = get_lab_range(metric_code, range_type)
    if base_range is None:
        return None
    
    info = LAB_RANGES[metric_code]
    min_val, max_val = base_range
    
    # Apply sex-specific adjustments based on notes
    if 'Creatinine' in info['name']:
        if sex == 'M':
            return (0.7, 1.3)
        elif sex == 'F':
            return (0.6, 1.1)
    
    elif 'Uric Acid' in info['name']:
        if sex == 'M':
            return (3.5, 7.2)
        elif sex == 'F':
            return (2.6, 6.0)
    
    elif 'HDL' in info['name']:
        if sex == 'F':
            return (50, 80)
        elif sex == 'M':
            return (40, 60)
    
    return (min_val, max_val)


def generate_realistic_baseline(metric_code: str, sex: str = None, health_status: str = "healthy") -> float:
    """
    Generate a realistic baseline value for a lab metric using AI-informed approach.
    
    Selects from appropriate range based on health status, then adds realistic variation.
    
    Args:
        metric_code: Lab metric code
        sex: 'M' or 'F' for sex-specific ranges
        health_status: 'healthy' (normal_range), 'clinical' (clinical_range), or 'extreme' (physiological_limit)
    
    Returns:
        Realistic baseline value
    """
    import numpy as np
    
    if metric_code not in LAB_RANGES:
        return 0.0
    
    info = LAB_RANGES[metric_code]
    
    # Select range based on health status
    if health_status == "healthy" and info['normal_range']:
        min_val, max_val = info['normal_range']
        # Use sex-adjusted range if available
        if sex:
            sex_range = get_sex_adjusted_range(metric_code, sex, "normal_range")
            if sex_range:
                min_val, max_val = sex_range
    elif health_status == "clinical" and info['clinical_range']:
        min_val, max_val = info['clinical_range']
    elif info['physiological_limit']:
        min_val, max_val = info['physiological_limit']
    else:
        return 0.0
    
    # Generate value with realistic distribution (normal distribution centered in range)
    center = (min_val + max_val) / 2
    std = (max_val - min_val) / 3
    
    value = np.random.normal(center, std)
    value = max(min_val, min(value, max_val))
    
    # Ensure positive if required
    if metric_code in CRITICAL_RULES["MUST_BE_POSITIVE"]:
        value = max(0.01, value)
    
    return float(value)


if __name__ == "__main__":
    print("=" * 80)
    print("HUMAN PHYSIOLOGICAL RANGES - ALL POPULATIONS")
    print("=" * 80)
    print()
    
    for code, info in sorted(LAB_RANGES.items()):
        if info['normal_range'] is None:
            continue
            
        print(f"{code}: {info['name']}")
        print(f"  Unit: {info['unit']}")
        print(f"  Normal (95% population): {info['normal_range'][0]}-{info['normal_range'][1]}")
        print(f"  Clinical (seen in practice): {info['clinical_range'][0]}-{info['clinical_range'][1]}")
        print(f"  Physiological limit: {info['physiological_limit'][0]}-{info['physiological_limit'][1]}")
        print(f"  Sex difference: {info['sex_difference']}")
        print(f"  Notes: {info['notes']}")
        print()

