# The finite-group invariant dataset

`data/group_properties_full.jsonl` is a catalogue of structural, subgroup-theoretic,
and representation-theoretic invariants for every finite group of order 21 to 255.
It is one JSON object per line, 6,958 lines, 153 columns each. It was built to drive
group selection and the clean-pair screen for this project, but it stands on its own
as a reference table, so this document describes it well enough to use without reading
any of the surrounding code.

| | |
| --- | --- |
| Rows | 6,958, one per group. Every order in 21–255 is present, with no gaps. |
| Key | `(order, index)`, the position in GAP's `SmallGroup` catalogue. `label` is the same pair as a string, e.g. `"128.2328"`, and is unique across the file, so it works as a primary key. |
| Columns | 153, identical set and order on every row. |
| sha256 | `b3071828c7d13eceb90453ad24c87e89adf402784f4a3c11d1d77e7e34b0536b` |

## How it was generated

A two-stage pipeline, both stages runnable from the repository:

```bash
gap -b -q -T scripts/enumerate_groups.g > data/group_properties.jsonl
uv run python scripts/enumerate_groups.py --output data/group_properties_full.jsonl
```

Stage 1 ([`scripts/enumerate_groups.g`](../scripts/enumerate_groups.g)) is a GAP
script that loops over `SmallGroup(order, index)` and writes the base invariants to an
immutable source catalogue. Stage 2
([`scripts/enumerate_groups.py`](../scripts/enumerate_groups.py), Sage/libgap,
available via `uv sync --extra sage`) reads that catalogue, computes the remaining
columns, repairs a handful the GAP stage left null, and writes the enriched file. Stage
2 refuses to run if `--output` resolves to its own input, so it cannot clobber the
source. Every column in stage 2 is assigned through an audit that fails the run loudly
if a column comes out entirely null without being declared sparse, so a silently dead
column cannot ship.

The two stages overlap: for a handful of invariants stage 2 recomputes what stage 1
already wrote (and its value wins in the file). The recomputed set is listed in
`ALGORITHMIC_CORE_FIELDS` in the stage-2 script; everything else is carried through from
GAP verbatim.

## Reading the schema

Types below are the JSON types you get from `json.loads`:

- `int`, `float`, `bool`, `str` are scalars. A few numeric columns
  (`average_character_degree`, `median_character_degree`) serialise as a bare `int`
  when the value is a whole number and `float` otherwise, because they come from
  Python's `statistics` functions; read them as numbers.
- `list` columns hold homogeneous lists; the element type is given in the description.
- `dict` columns are histograms or per-prime maps with **string** keys (JSON object
  keys are always strings, so a prime `3` appears as `"3"`).
- Sixteen columns can be `null`. A `null` is always a mathematical answer — "this
  property does not apply to this group" — never a missing measurement. The
  [Nullable columns](#nullable-columns) table below says what each one means.

## Column reference

### Identity

| Column | Type | Description |
| --- | --- | --- |
| `order` | int | The group order \|G\|, 21–255. |
| `index` | int | Catalogue index within the order, `1..NrSmallGroups(order)`. Largest is 2328 (order 128). |
| `label` | str | `"{order}.{index}"`, e.g. `"21.1"`. Unique per row. |
| `name` | str | GAP `StructureDescription(G)`, e.g. `"C7 : C3"`. A readable label: isomorphic groups always share it, but it can also collide on non-isomorphic ones, so treat it as a description rather than a key. |

### Classification flags

All boolean. Each is the corresponding GAP predicate unless noted.

| Column | Type | Description |
| --- | --- | --- |
| `abelian` | bool | `IsAbelian`. |
| `cyclic` | bool | `IsCyclic`. |
| `nilpotent` | bool | `IsNilpotentGroup`. |
| `solvable` | bool | `IsSolvableGroup`. 14 groups in range are non-solvable. |
| `supersolvable` | bool | `IsSupersolvableGroup`. |
| `monomial` | bool | `IsMonomialGroup` — every irreducible is induced from a linear character of a subgroup. |
| `metabelian` | bool | G'' = 1, tested as `Size(DerivedSubgroup(DerivedSubgroup(G))) == 1`. Computed directly rather than from `derived_length`, which is unreliable on non-solvable input. |
| `metacyclic` | bool | Has a cyclic normal subgroup with cyclic quotient; tested directly (this GAP has no `IsMetacyclicGroup`). |
| `perfect` | bool | `IsPerfectGroup`, G = G'. |
| `simple` | bool | `IsSimpleGroup`. |
| `is_nonabelian_simple` | bool | `IsNonabelianSimpleGroup`. |
| `is_almost_simple` | bool | `IsAlmostSimpleGroup`. |
| `is_quasisimple` | bool | `IsQuasisimpleGroup`. |
| `is_elementary_abelian` | bool | `IsElementaryAbelian`. |
| `is_p_group` | bool | `IsPGroup`. 2,792 groups in range are p-groups. |
| `dihedral` | bool | `IsDihedralGroup` (this one does exist in this GAP, unlike its neighbours). |
| `rational` | bool | Every irreducible character is rational-valued, tested as `num_rational_conjugacy_classes == number_conjugacy_classes`. |
| `is_frobenius` | bool | G is a Frobenius group. Computed from the semiregularity characterisation directly, since this GAP has no `IsFrobeniusGroup`. |
| `is_semidirect` | bool | A normal `1 < N < G` with a complement exists; G = N : H. The series-free extension flag, and safe to use as a feature. Direct products count as semidirect (trivial action). |
| `central_product` | bool | G = A·C_G(A) for a proper normal A with `1 < A ∩ C_G(A)`. Inclusive definition — true for 2,602 groups and dominated by trivial cases. See `is_essential_central_product`. |
| `is_essential_central_product` | bool | `central_product` and directly indecomposable. The discriminating column (653 groups). |
| `wreath_product` | bool | Isomorphic to a wreath product B ≀ T, decided by forward-enumerating all in-range wreath products and matching `IdGroup`. 26 groups. |
| `directly_indecomposable` | bool | The direct decomposition is the single factor `[G]`. |
| `has_faithful_irrep` | bool | A faithful irreducible complex representation exists, i.e. the socle of Z(G) is cyclic. 2,274 groups. |
| `has_cyclic_subgroup_index_2` | bool | Has a cyclic normal subgroup of index 2 (the dihedral / quaternion / semidihedral / modular ambient case). |
| `nilpotency_class_at_most_2` | bool | G' ≤ Z(G). True for every abelian group. Replaces the old `class2`, which was misread as "class == 2". |
| `nilpotency_class_exactly_2` | bool | Nilpotency class exactly 2; abelian groups are false. |
| `all_sylow_cyclic` | bool | Every Sylow subgroup is cyclic (a Z-group). |
| `all_sylow_abelian` | bool | Every Sylow subgroup is abelian (an A-group). |

### Order and series scalars

| Column | Type | Description |
| --- | --- | --- |
| `exponent` | int | `Exponent(G)`, the lcm of element orders. |
| `abelian_invariants` | list[int] | `AbelianInvariants(G)`, the abelian invariants of the abelianisation G/G' (e.g. `[3]` for a group with G/G' ≅ C3). |
| `derived_length` | int / null | `DerivedLength(G)`; **null when G is not solvable** (GAP returns a meaningless finite value there, so it is suppressed). |
| `nilpotency_class` | int / null | `NilpotencyClassOfGroup(G)`; **null when G is not nilpotent**. |
| `composition_length` | int | Number of factors in a composition series. Constant across all groups of a fixed order — a function of \|G\| alone. |
| `pc_rank` | int / null | Length of a polycyclic generating sequence, `Length(Pcgs(G))`; **null when G is not solvable**. Equals Ω(\|G\|), so constant across a fixed order. |
| `log_order` | float | `math.log(|G|)`. Constant across a fixed order. |
| `number_conjugacy_classes` | int | `NrConjugacyClasses(G)`. |
| `ngens` | int | `Length(MinimalGeneratingSet(G))`, d(G). The genuine "number of coordinate generators", and unlike `pc_rank` it varies at fixed order. |
| `center_order` | int | `Size(Centre(G))`. |
| `commutator_size` | int | `Size(DerivedSubgroup(G))`, \|G'\|. |
| `frattini_factor_size` | int | \|G\| / \|Φ(G)\|, the order of the Frattini quotient. |
| `upper_central_series_length` | int | Length of the upper central series (0 for a group with trivial centre chain, up to the nilpotency class). |
| `chief_series_length` | int | Number of factors in `ChiefSeries(G)`. |
| `elementary_abelian_series_length` | int / null | Number of factors in `ElementaryAbelianSeries(G)`; **null when G is not solvable**. |
| `stabilizer_chain_depth` | int | Base length of a stabiliser chain of a faithful permutation image — the iterative-sifting depth. |

### p-group structure

Null for every group that is not a p-group (the first three) or not a class-2 p-group
(the last two).

| Column | Type | Description |
| --- | --- | --- |
| `prime_p_group` | int / null | The prime p, `PrimePGroup(G)`; **null when G is not a p-group**. |
| `rank_p_group` | int / null | `RankPGroup(G)`, the rank of the Frattini quotient; **null when not a p-group**. |
| `p_class_p_group` | int / null | `PClassPGroup(G)`, the lower p-central length; **null when not a p-group**. |
| `commutator_form_rank` | int / null | Rank of the alternating commutator form on the Frattini quotient G/Φ(G), quotienting out generators that act centrally; **null unless G is a class-2 p-group**. |
| `commutator_form_radical_order` | int / null | Order of that form's radical, (Z(G)Φ(G))/Φ(G); **null unless G is a class-2 p-group**. |

### Automorphism group

Built once per group with `AutomorphismGroup(G)`.

| Column | Type | Description |
| --- | --- | --- |
| `aut_order` | int | \|Aut(G)\|. |
| `aut_order_ratio` | float | \|Aut(G)\| / \|G\|. |
| `aut_derived_length` | int / null | `DerivedLength(Aut(G))`; **null when Aut(G) is not solvable** (213 groups). Derived length is undefined for non-solvable groups — GAP returns a meaningless finite value there — so it is suppressed, exactly as `derived_length`. |
| `aut_solvable` | bool | `IsSolvableGroup(Aut(G))`. |
| `aut_nilpotent` | bool | `IsNilpotentGroup(Aut(G))`. |
| `aut_nilpotency_class` | int / null | `NilpotencyClassOfGroup(Aut(G))`; **null when Aut(G) is not nilpotent**. |
| `aut_nr_conjugacy_classes` | int | `NrConjugacyClasses(Aut(G))`. |
| `aut_exponent` | int | `Exponent(Aut(G))`. |

### Subgroup lattice

| Column | Type | Description |
| --- | --- | --- |
| `number_subgroups` | int | `Length(AllSubgroups(G))`, all subgroups, not conjugacy classes of them. |
| `number_normal_subgroups` | int | `Length(NormalSubgroups(G))`. |
| `subgroup_conjugacy_class_count` | int | Number of conjugacy classes of subgroups, from the subgroup lattice. |
| `normal_subgroup_orders` | list[int] | Orders of all normal subgroups. |
| `num_normal_subgroups_prime_order` | int | How many normal subgroups have prime order. |
| `maximal_subgroup_orders` | list[int] | Orders of all maximal subgroups (subgroups, not classes). |
| `total_maximal_subgroups` | int | Count of maximal subgroups. |
| `num_conjugacy_classes_maximal` | int | Number of conjugacy classes of maximal subgroups — not derivable from `total_maximal_subgroups`. |
| `num_minimal_normal_subgroups` | int | Count of minimal normal subgroups. |
| `minimal_normal_subgroup_orders` | list[int] | Their orders. |
| `normal_quotient_index_spectrum` | list[int] | Sorted distinct indices \|G:N\| over normal N (quotient orders, isomorphism types deliberately not retained). |

### Characteristic subgroups and radicals

All are the order (`Size`) of a canonical subgroup.

| Column | Type | Description |
| --- | --- | --- |
| `frattini_subgroup_order` | int | \|Φ(G)\|, intersection of the maximal subgroups. |
| `fitting_subgroup_order` | int | \|F(G)\|, the largest nilpotent normal subgroup. |
| `solvable_radical_order` | int | Order of the largest solvable normal subgroup. |
| `socle_order` | int | \|Soc(G)\|, product of the minimal normal subgroups. |
| `perfect_residuum_order` | int | Order of the perfect residuum (the last term of the derived series). |
| `supersolvable_residuum_order` | int | Order of the supersolvable residuum. |
| `p_core_orders` | list[int] | \|O_p(G)\| for each prime p dividing \|G\|, in ascending prime order (aligned with `sylow_subgroup_orders`). |
| `hypercenter_order` | int | Order of the hypercentre (first term of the upper central series). |

### Sylow structure

Per-prime lists, ascending in the prime, so index i is the same prime across
`sylow_numbers`, `sylow_subgroup_orders`, and `p_core_orders`.

| Column | Type | Description |
| --- | --- | --- |
| `sylow_numbers` | list[int] | n_p = \|G\| / \|N_G(P)\|, the number of Sylow p-subgroups, per prime. |
| `sylow_subgroup_orders` | list[int] | \|P\| for a Sylow p-subgroup, per prime. |
| `all_sylow_cyclic` | bool | (Also under classification flags.) |
| `all_sylow_abelian` | bool | (Also under classification flags.) |

### Chief series and direct decomposition

| Column | Type | Description |
| --- | --- | --- |
| `chief_factor_orders` | list[int] | Orders of the factors of `ChiefSeries(G)`, top to bottom. |
| `chief_factor_central` | list[bool] | Per factor, whether it is central in the quotient above it ([G, N_i] ≤ N_{i+1}). |
| `chief_factor_split` | list[bool] | Per factor, whether the extension splits. **Diagnostic only — not a group invariant**: it depends on which chief series GAP chose, and different chief series can disagree. Do not use as a feature; use `is_semidirect`. |
| `direct_factor_orders` | list[int] | Sorted orders of the directly indecomposable factors; `[|G|]` when G is itself indecomposable. |
| `direct_factor_count` | int | Number of those factors. |

### Frobenius and dihedral families

| Column | Type | Description |
| --- | --- | --- |
| `frobenius_kernel_order` | int / null | \|K\| of the Frobenius kernel; **null when G is not a Frobenius group** (134 groups are). |
| `frobenius_complement_order` | int / null | \|G\| / \|K\|; **null when not Frobenius**. |
| `frobenius_complement_is_cyclic` | bool / null | Whether the complement is cyclic (the pure affine "ax+b" case); **null when not Frobenius**. |
| `dihedral_family_type` | str | For a group with a cyclic index-2 subgroup, which twist family it is: `"dihedral"`, `"dicyclic_or_generalized_quaternion"`, `"semidihedral"`, `"modular"`, `"other_cyclic_index_2"`, or `"none"`. Distinguishes the four classical families that share an order and often a character table. |

### Element order and power structure

| Column | Type | Description |
| --- | --- | --- |
| `element_order_spectrum` | list[int] | Sorted distinct element orders. |
| `element_order_histogram` | dict[str,int] | Count of elements of each order, keyed by the order as a string. |
| `max_element_order` | int | Largest element order (= `exponent` for these groups). |
| `num_involutions` | int | Number of order-2 elements. |
| `fraction_prime_order` | float | Fraction of elements whose order is prime. |
| `element_order_entropy` | float | Shannon entropy (nats) of the element-order histogram. |
| `power_map_image_fraction` | dict[str,float] | For each prime p dividing \|G\|, the fraction of distinct images of x ↦ x^p, keyed by prime. |
| `power_map_fibre_histogram` | dict[str,dict] | For each prime p, the fibre-size distribution of x ↦ x^p (map from fibre size to how many images have it). |

### Conjugacy and commutator structure

| Column | Type | Description |
| --- | --- | --- |
| `conjugacy_class_sizes` | list[int] | Sizes of the conjugacy classes, in **GAP's class order (unsorted)** — see gotchas. |
| `max_conjugacy_class_size` | int | Largest class size. |
| `num_real_conjugacy_classes` | int | Classes equal to their own inverse class. |
| `num_rational_conjugacy_classes` | int | `Length(RationalClasses(G))`. |
| `commuting_probability` | float | k(G)/\|G\|, ratio of conjugacy classes to order — the probability two random elements commute. |
| `commutator_image_size` | int | Number of distinct commutators [x,y] (the image of the commutator map, not \|G'\|). |
| `commutator_fibre_histogram` | dict[str,int] | Fibre-size distribution of the commutator map. |
| `commutator_surjectivity_ratio` | float | `commutator_image_size` / \|G'\|. Exactly 1 for class-2 groups; below 1 certifies class ≥ 3. |

### Character degrees and representation theory

| Column | Type | Description |
| --- | --- | --- |
| `character_degrees` | list[int] | Degrees of the irreducible characters, in **GAP's `Irr` order (unsorted)** — see gotchas. |
| `max_irrep_dim` | int | Largest irreducible degree. |
| `distinct_degree_count` | int | Number of distinct irreducible degrees. |
| `average_character_degree` | number | Mean irreducible degree. |
| `median_character_degree` | number | Median irreducible degree. |
| `stddev_character_degree` | float | **Population** standard deviation of the degrees (÷n): the degrees are a complete population, so the sample estimator ÷(n−1) would be the wrong one. 0 when all degrees are equal. |
| `linC_count` | int | Number of linear (degree-1) irreducible characters = \|G/G'\|. |
| `linR_count` | int | Number of real-valued linear characters. |
| `fs_real_count` | int | Irreducibles with Frobenius–Schur indicator +1 (real). |
| `fs_complex_count` | int | Indicator 0 (complex). |
| `fs_quaternionic_count` | int | Indicator −1 (quaternionic). |
| `indicator_vector` | list[int] | The Frobenius–Schur indicator (−1/0/+1) per irreducible, in `Irr` order. |
| `num_rational_characters` | int | Irreducibles all of whose values are rational. |
| `character_field_degrees` | list[int] | [Q(χ):Q] per irreducible, `DegreeOverPrimeField(Field(chi))`. |
| `max_character_field_degree` | int | The maximum of those. |
| `fourier_block_cost` | int | Σ d³ over irreducible degrees d — a **cube**, a proxy for the cost of working in the Fourier blocks (Σ d² is just \|G\|). |
| `fourier_block_cost_normalized` | float | `fourier_block_cost` / \|G\|; always ≥ 1. |
| `plancherel_entropy` | float | Shannon entropy (nats) of the Plancherel measure P(ρ) = d²/\|G\|. |
| `plancherel_max` | float | max d² / \|G\|, the largest Plancherel weight. |
| `max_fusion_multiplicity` | int | Largest tensor fusion coefficient N_ij^k = ⟨χ_i χ_j, χ_k⟩. |
| `multiplicity_free_tensor_fraction` | float | Fraction of ordered irreducible pairs whose tensor product is multiplicity-free. |
| `mean_tensor_support` | float | Mean number of irreducible constituents in χ_i ⊗ χ_j over all i,j. |

### Faithful representations and actions

| Column | Type | Description |
| --- | --- | --- |
| `min_faithful_irrep_degree` | int / null | Smallest degree of a faithful irreducible complex representation; **null when none exists** (Z(G)'s socle is non-cyclic). |
| `irrR_degree` | int / null | Smallest degree of a faithful irreducible **real** representation (indicator +1 keeps the degree, 0 or −1 doubles it); **null under the same condition**. |
| `low_dim_irrep_kernel_profile` | list[list[int]] | For the five smallest irreducible degrees, `[degree, smallest kernel order at that degree]` — which quotients still have a cheap representation. |
| `min_faithful_rep_degree_sum` | int | Minimal total dimension of a faithful, possibly reducible, representation (exact subset cover of the minimal normal subgroups by irreducible kernels). Always defined, even when no faithful irreducible exists. |
| `min_faithful_rep_block_count` | int | Number of irreducible blocks in that minimal faithful representation. |
| `minimal_faithful_permutation_degree` | int | `MinimalFaithfulPermutationDegree(G)`. |
| `min_corefree_index` | int | Smallest index of a core-free subgroup — the minimal faithful *transitive* degree, which can exceed the permutation degree. |
| `corefree_index_spectrum` | list[int] | Sorted distinct indices of core-free subgroups. |
| `near_min_corefree_index_count` | int | How many entries of that spectrum are within 2× of the minimum. |
| `permutation_degree_ratio` | float | `minimal_faithful_permutation_degree` / \|G\|. |

### Character-table fingerprint, cohomology, isoclinism

| Column | Type | Description |
| --- | --- | --- |
| `character_table_fingerprint` | str | Canonical, permutation-invariant 24-hex hash of the ordinary character table. Equal fingerprint ⇔ equal character table up to simultaneous row/column permutation. Deliberately excludes power maps and Frobenius–Schur indicators (ν₂ is computed through the squaring power map, so it is not a table invariant). |
| `character_table_fingerprint_exact` | bool | False (350 groups) if the canonical-labelling search hit its node budget and fell back to a still-permutation-invariant refinement hash. Equal tables always collide either way, so a bucket is never wrongly split — only false merges are possible when this is false. |
| `schur_multiplier` | list[int] | Abelian invariants of the Schur multiplier H₂(G, ℤ), in GAP's ordering. `[]` means the trivial multiplier (order 1). Tried via the polycyclic route first, then `AbelianInvariantsMultiplier`, then HAP; raises rather than guessing if all fail. |
| `schur_multiplier_order` | int | Product of `schur_multiplier` (1 when the list is empty). |
| `isoclinism_family_id` | str | A **proxy** for isoclinism, not a certified class: a 24-hex hash of (\|G/Z\|, \|G'\|, commutator fibre histogram, exponent of G/Z). Groups sharing it are candidates for being isoclinic. |

### Derived fractions

Scalar ratios stored so downstream code does not recompute them. All in [0, 1] except
where noted.

| Column | Type | Description |
| --- | --- | --- |
| `centre_fraction` | float | \|Z(G)\| / \|G\|. |
| `abelianisation_fraction` | float | \|G/G'\| / \|G\| = 1 / \|G'\|. |
| `fitting_fraction` | float | \|F(G)\| / \|G\|. |
| `involution_fraction` | float | `num_involutions` / \|G\|. |
| `fs_complex_fraction` | float | `fs_complex_count` / k(G). |
| `fs_quaternionic_fraction` | float | `fs_quaternionic_count` / k(G). |
| `self_dual_fraction` | float | (`fs_real_count` + `fs_quaternionic_count`) / k(G), the fraction of real-valued (self-dual) irreducibles. |
| `rational_character_fraction` | float | `num_rational_characters` / k(G). |

## Nullable columns

Sixteen columns can be `null`, always meaning "not applicable to this group", never a
gap in the data. Grouped by the condition that produces the null:

| Condition | Columns | Count of null rows |
| --- | --- | --- |
| G is not a p-group | `prime_p_group`, `rank_p_group`, `p_class_p_group` | 4,166 |
| G is not solvable | `derived_length`, `pc_rank`, `elementary_abelian_series_length` | 14 |
| G is not nilpotent | `nilpotency_class` | 3,278 |
| Aut(G) is not solvable | `aut_derived_length` | 213 |
| Aut(G) is not nilpotent | `aut_nilpotency_class` | 4,011 |
| G is not a Frobenius group | `frobenius_kernel_order`, `frobenius_complement_order`, `frobenius_complement_is_cyclic` | 6,824 |
| No faithful irreducible complex rep exists | `min_faithful_irrep_degree`, `irrR_degree` | 4,684 |
| G is not a class-2 p-group | `commutator_form_rank`, `commutator_form_radical_order` | 5,830 |

## Gotchas

**`character_degrees` and `conjugacy_class_sizes` are stored unsorted.** They are in
GAP's internal `Irr` / conjugacy-class order, which is not canonical and differs between
groups. `conjugacy_class_sizes` is out of sorted order on 6,495 of the 6,958 rows and
`character_degrees` on 1,215. Sort both before comparing across groups or hashing them —
comparing them raw once silently matched about 2% of a target pool. `indicator_vector`
and `character_field_degrees` are in the same `Irr` order, so if you sort degrees for a
comparison, sort these alongside or not at all.

**Three columns are constant across every group of a fixed order.** `composition_length`,
`pc_rank`, and `log_order` are functions of \|G\| alone, so they carry no signal for
comparing groups of the same order. The within-order "number of generators" is `ngens`.

**`chief_factor_split` is not a group invariant.** It reads off one chief series that GAP
happened to pick, and a group has many. Use `is_semidirect` for the genuine
"does a complement exist" question.

**`central_product` is the inclusive definition** and holds for 2,602 groups, mostly
trivially. `is_essential_central_product` (central and directly indecomposable, 653
groups) is the one that discriminates.

**`isoclinism_family_id` is a proxy hash, not a certified isoclinism class.** Two groups
with the same id are candidates for being isoclinic, not a proof.

**`stddev_character_degree` is the population estimator** (÷n). It divides by the number
of irreducibles, because the degrees are the whole population, not a sample; the sample
estimator (÷(n−1)) would disagree with the enumerator.

**`fourier_block_cost` is a cube** (Σ d³), not the sum of squares. Σ d² is just \|G\|.

**Dict keys are strings.** `element_order_histogram`, `power_map_image_fraction`,
`power_map_fibre_histogram`, and `commutator_fibre_histogram` key on stringified
integers, so a prime `2` is the key `"2"`.

## Regenerating and verifying

Run the two commands under [How it was generated](#how-it-was-generated). Stage 1 needs
GAP with the `ctbllib` package; stage 2 needs Sage/libgap (with the `polycyclic` and,
ideally, `HAP` packages for the Schur multiplier). The column semantics that are easiest
to misread are also documented in the module docstring at the top of
[`scripts/enumerate_groups.py`](../scripts/enumerate_groups.py). The reproducibility
contract lives in [`docs/reproducibility.md`](reproducibility.md) and the study framing
in [`docs/methodology.md`](methodology.md).
