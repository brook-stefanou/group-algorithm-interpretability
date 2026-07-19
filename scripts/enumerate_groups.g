##############################################################################
#  enumerate_groups.g — GAP enumeration of SmallGroup invariants (STAGE 1)
#
#  Two-stage pipeline:
#    Stage 1 (this script): GAP enumerates SmallGroup invariants and writes
#      the immutable, regenerable source catalogue below.
#    Stage 2: scripts/enumerate_groups.py reads that stage-1 catalogue via
#      --source-group-properties (defaults to this file's path) and writes
#      the enriched canonical dataset to data/group_properties_full.jsonl via
#      --output. The two paths must differ; enumerate_groups.py refuses to
#      run if --output would overwrite this stage-1 file.
#
#  Run:  gap -b -q -T scripts/enumerate_groups.g > data/group_properties.jsonl
#
#  Loops over orders 21–255, computes structural and representation-theoretic
#  invariants for every SmallGroup, and writes one JSON line per group to
#  stdout.
#
#  Progress markers (lines starting with "#") are printed to stderr so they
#  don't pollute the JSONL output.
##############################################################################

LoadPackage("ctbllib");

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ORDER_MIN := 21;
ORDER_MAX := 255;
PROGRESS_INTERVAL := 50;  # print a progress marker every N groups

# ---------------------------------------------------------------------------
# Helper — convert a GAP boolean to JSON boolean
# ---------------------------------------------------------------------------
JSONBool := function(val)
    if val = true then
        return "true";
    else
        return "false";
    fi;
end;

# ---------------------------------------------------------------------------
# Helper — convert a GAP integer or fail to JSON null / integer
# ---------------------------------------------------------------------------
JSONInt := function(val)
    if val = fail then
        return "null";
    else
        return String(val);
    fi;
end;

# ---------------------------------------------------------------------------
# Helper — JSON string escaping
# ---------------------------------------------------------------------------
JSONString := function(s)
    local ch, result, c;
    result := "\"";
    for ch in [1..Length(s)] do
        c := s[ch];
        if c = '\n' then
            Append(result, "\\n");
        elif c = '\r' then
            Append(result, "\\r");
        elif c = '\t' then
            Append(result, "\\t");
        elif c = '\"' then
            Append(result, "\\\"");
        elif c = '\\' then
            Append(result, "\\\\");
        else
            Add(result, c);
        fi;
    od;
    Append(result, "\"");
    return result;
end;

# ---------------------------------------------------------------------------
# Safe wrappers that catch errors and return fail
#
# NOTE: GAP ``Tester`` filters check whether the attribute is already
# *cached* in the object's knowledge base, not whether it is *computable*.
# For a freshly constructed SmallGroup almost nothing is cached, so the
# testers would reject nearly everything.  We call the attribute directly
# and check the result type instead.
# ---------------------------------------------------------------------------
SafeIntAttribute := function(G, attr_name)
    local val;
    val := CallFuncList(ValueGlobal(attr_name), [G]);
    if IsInt(val) then
        return val;
    fi;
    return fail;
end;

SafeBoolAttribute := function(G, attr_name)
    local val;
    val := CallFuncList(ValueGlobal(attr_name), [G]);
    if val = true or val = false then
        return val = true;
    fi;
    return fail;
end;

# ---------------------------------------------------------------------------
# Main enumeration function for a single group
# ---------------------------------------------------------------------------
EnumerateGroup := function(order, index)
    local G, props, name, centre, derived, aut, cd, irr, ct, indicators,
          real_count, complex_count, quat_count, i, ind, cl, cs, subs, nsubs,
          primes, syl, n_p_list, order_list, sylow_cyclic, sylow_abelian,
          max_subs, max_orders, mins, lat;

    G := SmallGroup(order, index);
    props := rec();

    # --- Identity ---
    props.order := order;
    props.index := index;
    props.label := Concatenation(String(order), ".", String(index));
    props.name := StructureDescription(G);

    # --- Booleans ---
    props.abelian := IsAbelian(G);
    props.cyclic := IsCyclic(G);
    props.nilpotent := IsNilpotentGroup(G);
    props.solvable := IsSolvableGroup(G);
    props.simple := IsSimpleGroup(G);
    props.perfect := IsPerfectGroup(G);
    props.supersolvable := IsSupersolvableGroup(G);
    props.monomial := IsMonomialGroup(G);

    # --- Integer invariants ---
    props.exponent := SafeIntAttribute(G, "Exponent");
    props.derived_length := SafeIntAttribute(G, "DerivedLength");
    props.nilpotency_class := fail;
    if IsNilpotentGroup(G) then
        props.nilpotency_class := SafeIntAttribute(G, "NilpotencyClassOfGroup");
    fi;
    props.number_conjugacy_classes := SafeIntAttribute(G, "NrConjugacyClasses");

    # --- Composition length ---
    props.composition_length := fail;
    cs := CompositionSeries(G);
    props.composition_length := Length(cs) - 1;

    # --- Center ---
    props.center_order := fail;
    centre := Centre(G);
    if IsGroup(centre) then
        props.center_order := Size(centre);
    fi;

    # --- Derived subgroup (commutator) ---
    props.commutator_size := fail;
    derived := DerivedSubgroup(G);
    if IsGroup(derived) then
        props.commutator_size := Size(derived);
    fi;

    # --- Automorphism group (extended B9) ---
    props.aut_order := fail;
    props.aut_derived_length := fail;
    props.aut_solvable := fail;
    props.aut_nilpotent := fail;
    props.aut_nr_conjugacy_classes := fail;
    props.aut_exponent := fail;
    props.aut_order_ratio := fail;
    aut := AutomorphismGroup(G);
    if IsGroup(aut) then
        props.aut_order := Size(aut);
        props.aut_derived_length := DerivedLength(aut);
        props.aut_solvable := IsSolvableGroup(aut);
        props.aut_nilpotent := IsNilpotentGroup(aut);
        props.aut_nr_conjugacy_classes := NrConjugacyClasses(aut);
        props.aut_exponent := Exponent(aut);
        props.aut_order_ratio := Float(Size(aut)) / Float(Size(G));
    fi;

    # --- Minimal generator count ---
    props.ngens := fail;
    props.ngens := Length(MinimalGeneratingSet(G));

    # --- Subgroup counts ---
    props.number_subgroups := fail;
    props.number_normal_subgroups := fail;
    subs := AllSubgroups(G);
    if IsList(subs) then
        props.number_subgroups := Length(subs);
    fi;
    nsubs := NormalSubgroups(G);
    if IsList(nsubs) then
        props.number_normal_subgroups := Length(nsubs);
    fi;

    # --- Character degrees ---
    props.max_irrep_dim := -1;
    props.linC_count := 0;
    props.distinct_degree_count := 0;
    cd := CharacterDegrees(G);
    if IsList(cd) then
        props.distinct_degree_count := Length(cd);
        props.max_irrep_dim := Maximum(List(cd, x -> x[1]));
        props.linC_count := Sum(List(Filtered(cd, x -> x[1] = 1), x -> x[2]));
    fi;

    # --- Frobenius-Schur indicators ---
    props.fs_real_count := fail;
    props.fs_complex_count := fail;
    props.fs_quaternionic_count := fail;
    ct := CharacterTable(G);
    if IsCharacterTable(ct) then
        indicators := Indicator(ct, 2);
        if IsList(indicators) then
            real_count := 0;
            complex_count := 0;
            quat_count := 0;
            for i in [1..Length(indicators)] do
                ind := indicators[i];
                if ind = 1 then
                    real_count := real_count + 1;
                elif ind = 0 then
                    complex_count := complex_count + 1;
                elif ind = -1 then
                    quat_count := quat_count + 1;
                fi;
            od;
            props.fs_real_count := real_count;
            props.fs_complex_count := complex_count;
            props.fs_quaternionic_count := quat_count;
        fi;

        # Extended character theory (B6)
        props.character_degrees := fail;
        props.average_character_degree := fail;
        props.stddev_character_degree := fail;
        props.median_character_degree := fail;
        props.num_rational_characters := fail;
        props.indicator_vector := fail;

        irr := Irr(ct);
        if IsList(irr) then
            degrees := List(irr, d -> d[1]);
            props.character_degrees := degrees;
            props.indicator_vector := indicators;

            n_deg := Length(degrees);
            if n_deg > 0 then
                mean_deg := Float(Sum(degrees)) / Float(n_deg);
                props.average_character_degree := mean_deg;
                sq_diff := Sum(List(degrees, d -> (Float(d) - mean_deg)^2));
                props.stddev_character_degree := Sqrt(sq_diff / Float(n_deg));
            fi;

            # Median
            if n_deg > 0 then
                sorted_degs := ShallowCopy(degrees);
                Sort(sorted_degs);
                if n_deg mod 2 = 1 then
                    props.median_character_degree := sorted_degs[(n_deg + 1) / 2];
                else
                    props.median_character_degree := Float(sorted_degs[n_deg / 2] + sorted_degs[n_deg / 2 + 1]) / 2.0;
                fi;
            fi;

            # Count rational characters
            rational_cnt := 0;
            for chi in irr do
                all_rational := true;
                cc := ConjugacyClasses(ct);
                for c in cc do
                    val := chi[c];
                    if not IsRat(Cyclotomics(val)) then
                        all_rational := false;
                        break;
                    fi;
                od;
                if all_rational then
                    rational_cnt := rational_cnt + 1;
                fi;
            od;
            props.num_rational_characters := rational_cnt;
        fi;
    fi;

    # --- Category A: Additional structural booleans ---
    props.is_elementary_abelian := SafeBoolAttribute(G, "IsElementaryAbelian");
    props.is_p_group := SafeBoolAttribute(G, "IsPGroup");
    props.is_almost_simple := SafeBoolAttribute(G, "IsAlmostSimpleGroup");
    props.is_quasisimple := SafeBoolAttribute(G, "IsQuasisimpleGroup");
    props.is_nonabelian_simple := SafeBoolAttribute(G, "IsNonabelianSimpleGroup");
    props.is_frobenius := SafeBoolAttribute(G, "IsFrobeniusGroup");

    # p-group specific
    props.prime_p_group := fail;
    props.rank_p_group := fail;
    props.p_class_p_group := fail;
    if IsPGroup(G) then
        props.prime_p_group := SafeIntAttribute(G, "PrimePGroup");
        props.rank_p_group := SafeIntAttribute(G, "RankPGroup");
        props.p_class_p_group := SafeIntAttribute(G, "PClassPGroup");
    fi;

    # Frattini factor size
    props.frattini_factor_size := fail;
    if IsGroup(G) then
        props.frattini_factor_size := Size(G) / Size(FrattiniSubgroup(G));
    fi;

    # Abelian invariants
    props.abelian_invariants := fail;
    if IsGroup(G) then
        props.abelian_invariants := AbelianInvariants(G);
    fi;

    # --- Category B1: Element order statistics ---
    props.element_order_spectrum := fail;
    props.max_element_order := fail;
    props.num_involutions := fail;
    props.fraction_prime_order := fail;
    if IsGroup(G) then
        props.element_order_spectrum := Set(List(Elements(G), Order));
        props.max_element_order := Maximum(List(Elements(G), Order));
        props.num_involutions := Number(Elements(G), x -> Order(x) = 2);
        props.fraction_prime_order := Float(Number(Elements(G), x -> IsPrimeInt(Order(x)))) / Float(Size(G));
    fi;

    # --- Category B2: Conjugacy class structure ---
    props.conjugacy_class_sizes := fail;
    props.max_conjugacy_class_size := fail;
    props.num_rational_conjugacy_classes := fail;
    if IsGroup(G) then
        props.conjugacy_class_sizes := List(ConjugacyClasses(G), Size);
        props.max_conjugacy_class_size := Maximum(List(ConjugacyClasses(G), Size));
        props.num_rational_conjugacy_classes := Length(RationalClasses(G));
    fi;

    # --- Category B3: Characteristic subgroups ---
    props.frattini_subgroup_order := fail;
    if IsGroup(G) then
        props.frattini_subgroup_order := Size(FrattiniSubgroup(G));
    fi;

    props.fitting_subgroup_order := fail;
    if IsGroup(G) then
        props.fitting_subgroup_order := Size(FittingSubgroup(G));
    fi;

    props.solvable_radical_order := fail;
    if IsGroup(G) then
        props.solvable_radical_order := Size(SolvableRadical(G));
    fi;

    props.socle_order := fail;
    if IsGroup(G) then
        props.socle_order := Size(Socle(G));
    fi;

    props.perfect_residuum_order := fail;
    if IsGroup(G) then
        props.perfect_residuum_order := Size(PerfectResiduum(G));
    fi;

    props.supersolvable_residuum_order := fail;
    if IsGroup(G) then
        props.supersolvable_residuum_order := Size(SupersolvableResiduum(G));
    fi;

    props.p_core_orders := fail;
    if IsGroup(G) then
        props.p_core_orders := List(PrimeDivisors(Size(G)), p -> Size(PCore(G, p)));
    fi;

    props.hypercenter_order := fail;
    if IsGroup(G) then
        props.hypercenter_order := Size(Hypercentre(G));
    fi;

    # --- Category B4: Series lengths ---
    props.upper_central_series_length := fail;
    if IsGroup(G) then
        props.upper_central_series_length := Length(UpperCentralSeriesOfGroup(G)) - 1;
    fi;

    props.chief_series_length := fail;
    if IsGroup(G) then
        props.chief_series_length := Length(ChiefSeries(G)) - 1;
    fi;

    props.elementary_abelian_series_length := fail;
    if IsGroup(G) and IsSolvableGroup(G) then
        props.elementary_abelian_series_length := Length(ElementaryAbelianSeries(G)) - 1;
    fi;

    # --- Category B5: Sylow structure ---
    props.sylow_numbers := fail;
    props.sylow_subgroup_orders := fail;
    if IsGroup(G) then
        primes := PrimeDivisors(Size(G));
        n_p_list := [];
        order_list := [];
        for p in primes do
            syl := SylowSubgroup(G, p);
            Add(order_list, Size(syl));
            Add(n_p_list, Size(G) / Size(Normalizer(G, syl)));
        od;
        props.sylow_numbers := n_p_list;
        props.sylow_subgroup_orders := order_list;
    fi;

    props.all_sylow_cyclic := fail;
    if IsGroup(G) then
        props.all_sylow_cyclic := ForAll(PrimeDivisors(Size(G)), p -> IsCyclic(SylowSubgroup(G, p)));
    fi;

    props.all_sylow_abelian := fail;
    if IsGroup(G) then
        props.all_sylow_abelian := ForAll(PrimeDivisors(Size(G)), p -> IsAbelian(SylowSubgroup(G, p)));
    fi;

    # --- Category B7: Normal subgroups (extended) ---
    props.normal_subgroup_orders := fail;
    props.num_normal_subgroups_prime_order := fail;
    if IsGroup(G) then
        nsubs_list := NormalSubgroups(G);
        if IsList(nsubs_list) then
            ns_orders := List(nsubs_list, Size);
            props.normal_subgroup_orders := ns_orders;
            props.num_normal_subgroups_prime_order := Number(ns_orders, s -> IsPrimeInt(s));
        fi;
    fi;

    # --- Category B8: Maximal subgroups ---
    props.maximal_subgroup_orders := fail;
    props.total_maximal_subgroups := fail;
    if IsGroup(G) then
        max_subs := MaximalSubgroups(G);
        if IsList(max_subs) then
            max_orders := List(max_subs, Size);
            props.maximal_subgroup_orders := max_orders;
            props.total_maximal_subgroups := Length(max_subs);
        fi;
    fi;

    # --- Category B10-B12: Minimal normal subgroups, commuting prob, Frobenius, subgroup classes ---
    props.num_minimal_normal_subgroups := fail;
    props.minimal_normal_subgroup_orders := fail;
    if IsGroup(G) then
        mins := MinimalNormalSubgroups(G);
        if IsList(mins) then
            props.num_minimal_normal_subgroups := Length(mins);
            props.minimal_normal_subgroup_orders := List(mins, Size);
        fi;
    fi;

    props.commuting_probability := fail;
    if IsGroup(G) then
        props.commuting_probability := Float(NrConjugacyClasses(G)) / Float(Size(G));
    fi;

    props.frobenius_kernel_order := fail;
    props.frobenius_complement_order := fail;
    if IsGroup(G) and IsFrobeniusGroup(G) then
        props.frobenius_kernel_order := Size(FrobeniusKernel(G));
        props.frobenius_complement_order := Size(FrobeniusComplement(G));
    fi;

    props.subgroup_conjugacy_class_count := fail;
    if IsGroup(G) then
        lat := LatticeSubgroups(G);
        props.subgroup_conjugacy_class_count := Length(ConjugacyClassesSubgroups(lat));
    fi;

    return props;
end;

# ---------------------------------------------------------------------------
# JSON serialisation of a single group record
# ---------------------------------------------------------------------------
PropsToJSON := function(props)
    local parts, key, val;
    parts := [];

    for key in RecNames(props) do
        val := props.(key);
        Add(parts, Concatenation(JSONString(key), ": "));
        if IsBool(val) then
            Add(parts, JSONBool(val));
        elif IsInt(val) then
            Add(parts, JSONInt(val));
        elif IsString(val) then
            Add(parts, JSONString(val));
        elif val = fail then
            Add(parts, "null");
        elif IsFloat(val) then
            Add(parts, String(val));
        elif IsList(val) then
            parts_for_list := [];
            for item in val do
                if IsInt(item) then
                    Add(parts_for_list, String(item));
                elif IsBool(item) then
                    Add(parts_for_list, JSONBool(item));
                elif IsFloat(item) then
                    Add(parts_for_list, String(item));
                elif item = fail then
                    Add(parts_for_list, "null");
                else
                    Add(parts_for_list, JSONString(String(item)));
                fi;
            od;
            Add(parts, Concatenation("[", JoinStringsWithSeparator(parts_for_list, ", "), "]"));
        else
            Add(parts, JSONString(String(val)));
        fi;
        Add(parts, ", ");
    od;
    if Length(parts) > 0 then
        # Remove trailing ", "
        Unbind(parts[Length(parts)]);
        Unbind(parts[Length(parts)-1]);
    fi;

    # Build compact JSON object.  We do this by hand rather than importing a
    # JSON library that the vanilla GAP distribution might not have.
    return Concatenation("{", Concatenation(parts), "}");
end;

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
main := function()
    local order, index, n_groups, count, props, json_line, order_start;

    count := 0;

    # Print header marker to stderr
    PrintTo("*err*", "# enumerate_groups.g: orders ", ORDER_MIN, "-", ORDER_MAX, "\n");

    for order in [ORDER_MIN .. ORDER_MAX] do
        n_groups := NrSmallGroups(order);
        if n_groups = 0 then
            continue;
        fi;

        PrintTo("*err*", "#   Order ", order, " (", n_groups, " groups)...\n");

        for index in [1 .. n_groups] do
            count := count + 1;
            props := EnumerateGroup(order, index);

            # Output JSON line
            json_line := PropsToJSON(props);
            Print(json_line, "\n");

            # Progress
            if count mod PROGRESS_INTERVAL = 0 then
                PrintTo("*err*", "#   ... ", count, " groups done\n");
            fi;
        od;
    od;

    PrintTo("*err*", "# Total groups enumerated: ", count, "\n");
end;

main();
QUIT;
