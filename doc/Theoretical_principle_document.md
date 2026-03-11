# Low Thermal Conductivity Material Design - Theoretical Principles Document

---

## 1. Element Selection Theory

### 1.1 Definition of Element Library

**Allowed 14 Elements**:

| Category | Elements | Core Characteristics |
| :--- | :--- | :--- |
| **Transition Metals** | Ti, V, Cu, Ag | Multiple oxidation states, electrical conductivity modulation |
| **Main-group Metals** | In, Ge, Sn, Pb | Heavy atoms, lone pair electrons |
| **Metalloids** | As, Sb, Bi | Heavy atoms, strong lone pair electrons |
| **Chalcogens** | S, Se, Te | Mass gradient, covalency modulation |

### 1.2 Selection Principles

#### Theory 1: Heavy Atom Effect
- **Mechanism**: Phonon group velocity is proportional to the $1/\sqrt{M}$. Here $M$ is the mass.
- **Target Elements**: Pb, Bi, Sb, Te.
- **Effect**: Reduces lattice thermal conductivity.

#### Theory 2: Lone Pair Electron Effect
- **Mechanism**: Asymmetric distribution of $ns^2$ electrons enhances anharmonicity.
- **Active Ions**: Pb²⁺, Bi³⁺, Sb³⁺, Sn²⁺.
- **Effect**: Weakens chemical bonds, enhances phonon scattering and reduce phonon relaxation time.

#### Theory 3: Mass Contrast Scattering
- **Mechanism**: Large mass differences between atoms enhance scattering.
- **Strategy**: Heavy-Light element pairing (e.g., Pb-S, Bi-Te).
- **Parameter**: $\Gamma_M = \sum f_i(1 - M_i/M_{avg})^2$.

#### Theory 4: Known Material Families
- **Bi-Te System**: Layered structure.
- **Cu-Se System**: Superionic conductors.
- **Sn-Se System**: Strong anharmonicity.
- **Pb-Te System**: Classic thermoelectric materials.
- **Ag-Sb-Te System**: Complex structures.

---

## 2. Low Thermal Conductivity Design Theory

### 2.1 Basic Thermal Conductivity Equation

```
κ_L = (1/3) ∫ C_v(ω) · v_g²(ω) · τ(ω) dω
```

**Design Core**: Enhance phonon scattering, reduce phonon group velocity, and reduce relaxation time ($\tau$).

---

### 2.2 Four Major Phonon Scattering Mechanisms

#### Mechanism 1: Mass Contrast Scattering
- **Principle**: Mass differences scatter phonons.
- **Parameter**: $\Gamma_M = \sum f_i(1 - M_i/M_{avg})^2$.
- **Strategy**: Maximize mass difference increases the acoustic-optical phonon branches gap.

#### Mechanism 2: Lattice Distortion Scattering
- **Principle**: Atomic radius differences create strain fields.
- **Parameter**: $\Gamma_S = \sum f_i(1 - r_i/r_{avg})^2$.
- **Strategy**: Moderate mismatch, such as the (5%,15%) range.

#### Mechanism 3: Resonance Scattering
- **Principle**: Weak bonding between atoms resonates with phonons.
- **Applicable Structures**: Cage structures (Skutterudite, Clathrate).
- **Strategy**: Fill voids with heavy atoms ("Rattlers") generate flat bands in acoustic phonon modes.

#### Mechanism 4: Interface Scattering
- **Principle**: Interfaces scatter long-wavelength phonons.
- **Applicable Structures**: Layered structures, nanocomposites, superlattices.
- **Characteristics**: Anisotropic thermal conductivity.

---

### 2.3 Lattice Anharmonicity Theory

**Core**: Strong anharmonicity → Enhanced phonon-phonon scattering → Shortened phonon lifetime.

**Three Major Sources**:

1.  **Weak Chemical Bonds**
    - Van Der Waals bonds.
    - Weak interlayer interactions in layered structures.

2.  **Lone Pair Electrons**
    - Asymmetric distribution of $ns^2$ lone pairs.
    - Elements: Pb²⁺, Bi³⁺, Sb³⁺.

3.  **Low Symmetry Structures**
    - Reduces phonon group velocity.
    - Increases Phonon Density of States (PDOS).

---

### 2.4 Complex Structure Effects

**Principle**: Complex structure → Low phonon group velocity.

**Structure Types**:
- **Multi-atom Primitive Cell**: Large number of atoms per unit cell.
- **Layered Structure**: Anisotropy.
- **Cage Structure**: Resonance scattering.
- **Superlattice**: Periodic interfaces.

---

## 3. Crystal Stability Criteria

### 3.1 Thermal Dynamical Stability
- **Criterion**: $\omega^2(q) > 0$ (No imaginary frequencies in the phonon spectrum).
- **Significance**: Stable lattice vibrations.

### 3.2 Thermodynamic Stability
- **Criterion**: $\Delta H_f < 0$ (Negative formation energy).
- **Significance**: Material does not spontaneously decompose.

### 3.3 Mechanical Stability
- **Criterion**: Elastic constants satisfy Born criteria.
- **Significance**: Material possesses resistance to deformation.

---

## 4. Success Case Library (Dynamic Update Zone)

> **Update Mechanism Explanation**:
> - This section records material cases verified by experiments or AI screening.
> - After each iteration, update according to the following rules:
>   1.  **New Cases**: Materials experimentally verified with thermal conductivity < threshold.
>   2.  **Case Format**: Chemical Formula | Thermal Conductivity | Key Theoretical Features.
>   3.  **Theory Extraction**: Extract new patterns from cases and feed back into Sections 1-3.
>   4.  **Weight Adjustment**: Prioritize high-frequency successful features.

### 4.1 Initial Reference Cases (Literature Benchmarks)

Theoretical features of successful materials based on literature:

- **Bi₂Te₃** ($\kappa = 0.87$ W/m·K): Layered structure + Lone pair electrons + Heavy atoms.
- **SnSe** ($\kappa = 0.73$ W/m·K): Layered structure + Lone pair electrons + Low symmetry.
- **Cu₂Se** ($\kappa = 0.67$ W/m·K): Superionic conductor + Mass contrast.

### 4.2 Success Case Theory Summary (Iterative Update)

> **Update Mechanism**: After each iteration, extract common patterns from verified successful materials and update this section as a theoretical summary.

**Current Status**: Initial version, pending update after the first iteration.

**Expected Content Examples**:
- Element Combination Patterns: Certain element pairs have high success rates.
- Mass Contrast Range: Distribution characteristics of $\Gamma_M$ in successful materials.
- Lone Pair Effect: Relationship between the proportion of lone pair elements and success rate.
- Structural Features: Common crystal structure characteristics of successful materials.

### 4.3 Theory Optimization Record

> **Update Mechanism**: Record theoretical adjustments based on case feedback.

**Current Status**: Initial version, pending update after the first iteration.

**Expected Content Examples**:
- Discovery: Certain element combination pattern → Adjust element priority.
- Discovery: Certain structural feature → Update screening weights.
- Discovery: Certain failure mode → Add exclusion rules.

---

## 5. Material Evaluation Core Rules

### 5.1 Core Evaluation Parameters

1.  **Mass Contrast** ($\Gamma_M$): The larger, the better.
2.  **Lone Pair Electrons**: Prioritize Pb, Bi, Sb, Sn.
3.  **Average Mass**: Preference for heavy atoms.
4.  **Electronegativity Difference**: for example, $0.4 < \Delta\chi < 2.0$.

### 5.2 Screening Decision

```
Lone pair element + High mass contrast → Recommended
Lone pair element OR Cu/Ag → Moderate
Others → Low Potential
```

### 5.3 Confidence Assessment

- **High Confidence**: Belongs to known family + Lone pair electrons + High mass contrast.
- **Moderate Confidence**: Fits partial features + New element combination.
- **Low Confidence**: Theoretical features not obvious + Stability questionable.

---

## 6. Core Design Principles Summary

### 6.1 Four Theoretical Pillars

1.  **Heavy Atom Effect** → Reduces phonon group velocity.
2.  **Lone Pair Electron Effect** → Enhances anharmonicity.
3.  **Mass Contrast Scattering** → Enhances phonon scattering.
4.  **Complex Structure** → Multiple scattering mechanisms.

### 6.2 Material Design Checklist

Check items one by one when evaluating materials:

- [ ] Contains heavy elements (Pb, Bi, Sb, Te).
- [ ] Contains lone pair electron elements.
- [ ] Significant mass contrast (High $\Gamma_M$).
- [ ] Belongs to a known low thermal conductivity family.
- [ ] Moderate structural complexity.
- [ ] Dynamically stable (No imaginary frequencies).
- [ ] Thermodynamically stable ($\Delta H_f < 0$).
- [ ] Reasonable electronegativity and radius matching.