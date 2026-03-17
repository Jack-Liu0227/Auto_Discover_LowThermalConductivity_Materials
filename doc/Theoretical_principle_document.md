# Low Thermal Conductivity Material Re-Ranking Principles

---

## 1. Optimization Goal

The purpose of this document is to support theory-guided re-ranking of a provided candidate pool.

Given candidate compositions together with predicted thermal conductivity and related model parameters, the task is:

1. Internally re-rank the candidate pool.
2. Retain only the top-N most promising materials.
3. Prioritize candidates that are more likely to combine:
   - low lattice thermal conductivity
   - dynamical stability
   - chemically reasonable bonding and structural motifs

This document is not a hard filter. It provides ranking signals for deciding which candidates should move into the final retained top-N list.

---

## 2. Element Selection Theory

### 2.1 Definition of Element Library

**Allowed 14 Elements**:

| Category | Elements | Core Characteristics |
| :--- | :--- | :--- |
| **Transition Metals** | Ti, V, Cu, Ag | Variable coordination, electronic flexibility |
| **Main-group Metals** | In, Ge, Sn, Pb | Moderate to high mass, soft bonding tendencies |
| **Metalloids** | As, Sb, Bi | Lone-pair activity, low-symmetry tendency |
| **Chalcogens** | S, Se, Te | Mass gradient, bonding diversity |

### 2.2 Selection Principles

#### Theory 1: Heavy Atom Effect
- **Mechanism**: Higher atomic mass lowers phonon group velocity.
- **Ranking signal**: Raise candidates containing Pb, Bi, Sb, and Te when mass increase does not introduce obvious stability risk.

#### Theory 2: Lone Pair Electron Effect
- **Mechanism**: Active ns^2 lone pairs increase lattice anharmonicity.
- **Ranking signal**: Raise candidates containing Pb2+, Sn2+, Sb3+, and Bi3+ when the composition suggests distorted or low-symmetry environments.

#### Theory 3: Mass Contrast Scattering
- **Mechanism**: Large mass differences broaden phonon scattering across frequencies.
- **Ranking signal**: Raise candidates with useful heavy-light contrast inside the same chemically plausible framework.

#### Theory 4: Known Material Families
- **Mechanism**: Proven low-kappa families offer reliable starting points for screening.
- **Ranking signal**: Treat Bi-Te, Cu-Se, Sn-Se, Pb-Te, and Ag-Sb-Te related systems as positive family-level priors, not automatic winners.

---

## 3. Low Thermal Conductivity Design Theory

### 3.1 Basic Thermal Conductivity Equation

```math
\kappa_L \approx \frac{1}{3} \int C_v(\omega) v_g^2(\omega) \tau(\omega)\, d\omega
```

**Design target**: Reduce phonon group velocity `v_g` and phonon lifetime `tau` while keeping the lattice stable.

### 3.2 Positive Ranking Signals

#### Signal 1: Mass Contrast Scattering
- **Principle**: Mass differences disrupt phonon transport.
- **Top-N cue**: Raise materials where heavy-light contrast is meaningful and chemically coherent.

#### Signal 2: Lattice Distortion Scattering
- **Principle**: Local strain fields scatter phonons.
- **Top-N cue**: Raise materials with moderate atomic-size mismatch and likely distorted coordination.

#### Signal 3: Resonance or Soft-Bond Scattering
- **Principle**: Weakly bound atoms or soft modes add localized scattering channels.
- **Top-N cue**: Raise materials that suggest soft sublattices, rattling motifs, or weak interlayer bonding without obvious instability.

#### Signal 4: Interface or Layer Scattering
- **Principle**: Structural interfaces suppress long-wavelength phonons.
- **Top-N cue**: Raise layered, anisotropic, or internally heterogeneous frameworks when supported by the composition and known family behavior.

### 3.3 Negative Ranking Signals

- Candidates that rely only on heavy elements but show no strong lone-pair, soft-bond, or structural-complexity support should not be ranked too aggressively.
- Candidates that look chemically rigid or overly covalent should be lowered unless other strong low-kappa signals are present.
- Candidates that strongly deviate from known successful families should be lowered when the mechanism support is weak.
- Structural complexity alone is not enough to enter the retained top-N list.

### 3.4 Lattice Anharmonicity Theory

**Core idea**: Strong anharmonicity shortens phonon lifetime and lowers lattice thermal conductivity.

**Primary sources**:
1. **Weak bonds**: Soft or weakly coupled bonds reduce restoring forces.
2. **Lone-pair electrons**: Pb2+, Sn2+, Sb3+, and Bi3+ often drive local asymmetry.
3. **Low symmetry**: Distorted lattices lower phonon coherence and raise scattering.

### 3.5 Complex Structure Effects

**Why it matters**: Complex, low-symmetry structures increase optical branches and reduce coherent heat transport.

**Favorable traits**:
- Multi-atom primitive cells
- Layered or anisotropic frameworks
- Distorted coordination environments

---

## 4. Stability Considerations

### 4.1 Dynamical Stability
- **Criterion**: No imaginary frequencies in the phonon spectrum.
- **Significance**: The lattice can sustain stable vibrations.
- **Ranking use**: Stability-supporting evidence should raise priority. Clear instability risk should lower priority even if the predicted thermal conductivity is attractive.

### 4.2 Thermodynamic Stability
- **Criterion**: Negative formation energy or a clearly competitive decomposition energy.
- **Significance**: The material is not strongly driven to decompose.
- **Ranking use**: Thermodynamic competitiveness is a positive signal for entering the retained top-N list.

### 4.3 Mechanical Stability
- **Criterion**: Elastic constants satisfy the relevant Born stability conditions.
- **Significance**: The structure can resist small mechanical perturbations.
- **Ranking use**: Use as a supporting stability signal when available.

---

## 5. Family-Level Patterns

- Ag-based chalcogenides remain important reference families.
- Sulfosalt-like systems containing Sb, Bi, As, Sn, or Pb deserve special attention when they also show soft-bond or lone-pair support.
- S-, Se-, and Te-based systems should not be ranked by chalcogen mass alone. The final rank should reflect the combined balance of lone-pair activity, mass contrast, structural complexity, and stability likelihood.

---

## 6. Selection Preference For Top-N

When only the final top-N materials will be retained, prefer candidates that satisfy multiple signals at once.

### 6.1 Strong Top-N Preference

- Meaningful low predicted thermal conductivity together with plausible stability.
- Heavy elements combined with lone-pair activity or soft-bond motifs.
- Family-level consistency with known low-kappa systems.
- Compositions suggesting low symmetry, distortion, or multi-atom complexity.

### 6.2 Medium Top-N Preference

- Candidates with one strong low-kappa mechanism and otherwise acceptable stability outlook.
- Candidates with good model-side parameters but only partial theory support.

### 6.3 Reasons To Fall Out Of Top-N

- Weak mechanism support despite a favorable predicted thermal conductivity.
- Clear stability concerns or chemically implausible bonding balance.
- Reliance on a single weak signal such as high average atomic mass alone.

---

## 7. Evidence Update Log

> This section is the dynamic update zone for successful cases and theory revisions.

### 7.1 Success Case Theory Summary (Iterative Update)

> Update this subsection with concise cross-case patterns extracted from verified successful materials.

**Current Status**: Initial version; no iterative evidence has been added yet.

**Template fields**:
- **Dominant element systems**:
- **Mass-contrast pattern**:
- **Lone-pair or bonding pattern**:
- **Structural pattern**:

### 7.2 Theory Optimization Record

> Update this subsection with theory changes that should affect later screening decisions.

**Current Status**: Initial version; no optimization adjustments have been recorded yet.

**Template fields**:
- **Priority adjustment**:
- **Screening filter adjustment**:
- **Exclusion or caution rule**:

---

## 8. Material Evaluation Core Rules

### 8.1 Core Evaluation Parameters

1. **Mass contrast** (`Gamma_M`): Larger is usually better.
2. **Lone-pair activity**: Prefer compounds containing Pb, Sn, Sb, or Bi when chemically reasonable.
3. **Average atomic mass**: Heavier frameworks are preferred when stability is preserved.
4. **Bonding balance**: Favor compositions with neither fully metallic nor overly rigid bonding.

### 8.2 Re-Ranking Decision

```text
Multiple strong low-kappa signals + acceptable stability outlook -> Move toward the top of the retained top-N list
Partial support with manageable risk -> Keep as mid-priority candidate
Weak mechanism support or notable stability risk -> Lower priority or drop from retained top-N
```

### 8.3 Output Rule

- First internally re-rank the provided candidate pool.
- Then retain only the requested top-N materials.
- Output only the retained materials.
- The final output order must be the final ranking order.

---

## 9. Core Design Principles Summary

### 9.1 Four Theoretical Pillars

1. **Heavy atom effect** -> lowers phonon group velocity.
2. **Lone-pair electron effect** -> increases anharmonicity.
3. **Mass-contrast scattering** -> broadens phonon scattering.
4. **Complex structure effect** -> reduces coherent heat transport.

### 9.2 Material Re-Ranking Checklist

Check the following when deciding whether a candidate should remain in the final retained top-N list:

- [ ] Contains heavy elements such as Pb, Bi, Sb, or Te
- [ ] Contains lone-pair-active cations when chemically plausible
- [ ] Provides meaningful heavy-light mass contrast
- [ ] Fits a known or adjacent low-kappa family
- [ ] Shows low symmetry or structural complexity
- [ ] Is dynamically stable
- [ ] Is thermodynamically competitive
- [ ] Has a chemically reasonable bonding environment
