##############################################################################
# Fourier-falsifier screen -- GAP library.
#
# Two entry points, each printing exactly one JSON line to stdout so the Python
# driver can parse them (via the same GAP-output repair used in i14/clean_ext.py:
# strip backslash-newline, collapse newline->space, "0." -> "0.0").
#
#   CTProof(o1,i1,o2,i2)  -- prove/refute ordinary-character-table equality of
#     two SmallGroups with TransformingPermutations(Irr(G),Irr(H)).  This is the
#     CORRECT test: it asks only for an order-preserving bijection of the
#     irreducible characters, NOT for a compatible power map.  NEVER use
#     TransformingPermutationsCharacterTables here -- that also compares power
#     maps and would falsely reject a genuinely CT-equal pair whose Frobenius-
#     Schur data (a power-map invariant) differs.
#
#   GroupData(o,i)  -- everything the screen needs about a single group:
#     * the Frobenius-Schur joint multiset  Collected([ chi(1), nu2(chi) ])
#       over Irr(G), with nu2 = Indicator(ct,2)  (this is the strict FS profile;
#       its projection to counts of +1/0/-1 is the loose r/c/q triple).
#     * the FS-involution identity  Sum_chi nu2(chi)*chi(1) == #{ g : g^2 = 1 },
#       computed both ways and asserted.
#     * for every conjugacy class of core-free subgroup H (Core(G,H)=1, H<G):
#       the decomposition template of Ind_H^G(1) -- the multiset of
#       [ chi(1), m_chi ] with m_chi = <Res_H chi, 1_H> > 0 -- together with the
#       index [G:H] and the support (# irreps with m_chi>0).  Frobenius
#       reciprocity  Sum m_chi*chi(1) == [G:H]  is asserted for every template.
#     * min_corefree_index and the sorted core-free index spectrum, as a GAP
#       cross-check of the same jsonl fields.
#
# Multiplicities are computed as m_chi = <Res_H chi, 1_H> (dimension of the
# H-fixed subspace); this equals <Ind_H^G 1, chi> by Frobenius reciprocity and
# needs no fusion/permutation-character machinery.
##############################################################################

# JSON-encode a GAP list of small integers as "[a, b, c]".
IntListJSON := function(xs)
    local parts, x;
    parts := List(xs, x -> String(x));
    return Concatenation("[", JoinStringsWithSeparator(parts, ", "), "]");
end;

# JSON-encode a multiset produced by Collected -- a list of [ item, count ]
# where item is itself a short integer list [a,b] -- as
#   [[[a, b], count], ...]
CollectedPairsJSON := function(coll)
    local parts, e;
    parts := [];
    for e in coll do
        Add(parts, Concatenation("[", IntListJSON(e[1]), ", ", String(e[2]), "]"));
    od;
    return Concatenation("[", JoinStringsWithSeparator(parts, ", "), "]");
end;

CTProof := function(o1, i1, o2, i2)
    local G, H, tp, ok;
    G := SmallGroup(o1, i1);
    H := SmallGroup(o2, i2);
    tp := TransformingPermutations(Irr(CharacterTable(G)), Irr(CharacterTable(H)));
    ok := tp <> fail;
    Print("{\"kind\": \"ctproof\"",
          ", \"a\": [", o1, ", ", i1, "]",
          ", \"b\": [", o2, ", ", i2, "]",
          ", \"ct_equal_irr\": ", String(ok),
          "}\n");
end;

GroupData := function(o, i)
    local G, n, ct, irr, k, degs, nu2, joint, jointJSON,
          rcount, ccount, qcount, fssum, ninv, fsok,
          ccs, reps, H, tH, res, m, tmpl, deg, mult, kk, idx, support,
          templates, tparts, spectrum, mci;

    G := SmallGroup(o, i);
    n := Size(G);
    ct := CharacterTable(G);
    irr := Irr(ct);
    k := Length(irr);
    degs := List(irr, x -> x[1]);
    nu2 := Indicator(ct, 2);

    # strict FS joint multiset {(deg, nu2)}
    joint := Collected(List([1 .. k], kk -> [degs[kk], nu2[kk]]));
    jointJSON := CollectedPairsJSON(joint);
    rcount := Number(nu2, x -> x = 1);
    ccount := Number(nu2, x -> x = 0);
    qcount := Number(nu2, x -> x = -1);

    # FS-involution identity: Sum nu2(chi)*chi(1) = #{g : g^2 = 1}
    fssum := Sum([1 .. k], kk -> nu2[kk] * degs[kk]);
    ninv := Number(G, g -> g^2 = One(G));
    fsok := fssum = ninv;
    if not fsok then
        Error("FS-involution identity FAILED for (", o, ",", i, "): ",
              fssum, " <> ", ninv);
    fi;

    # core-free subgroup templates
    ccs := ConjugacyClassesSubgroups(G);
    reps := List(ccs, Representative);
    templates := [];
    spectrum := [];
    for H in reps do
        if Size(H) < n and Size(Core(G, H)) = 1 then
            idx := n / Size(H);
            Add(spectrum, idx);
            tH := CharacterTable(H);
            res := List(irr, chi -> ScalarProduct(tH,
                        RestrictedClassFunction(chi, tH), TrivialCharacter(tH)));
            # Frobenius reciprocity check
            if Sum([1 .. k], kk -> res[kk] * degs[kk]) <> idx then
                Error("Frobenius reciprocity FAILED for (", o, ",", i,
                      ") H index ", idx);
            fi;
            tmpl := [];
            support := 0;
            for kk in [1 .. k] do
                if res[kk] > 0 then
                    Add(tmpl, [degs[kk], res[kk]]);
                    support := support + 1;
                fi;
            od;
            Sort(tmpl);
            Add(templates, rec(idx := idx, support := support,
                               tmpl := Collected(tmpl)));
        fi;
    od;
    Sort(spectrum);
    if Length(spectrum) = 0 then
        mci := n;
    else
        mci := spectrum[1];
    fi;

    tparts := List(templates, t -> Concatenation(
        "{\"idx\": ", String(t.idx),
        ", \"support\": ", String(t.support),
        ", \"tmpl\": ", CollectedPairsJSON(t.tmpl), "}"));

    Print("{\"kind\": \"groupdata\"",
          ", \"id\": [", o, ", ", i, "]",
          ", \"fs_joint\": ", jointJSON,
          ", \"fs_triple\": [", rcount, ", ", ccount, ", ", qcount, "]",
          ", \"fs_involution_sum\": ", fssum,
          ", \"num_involutions\": ", ninv,
          ", \"fs_involution_ok\": ", String(fsok),
          ", \"min_corefree_index\": ", mci,
          ", \"corefree_index_spectrum\": ", IntListJSON(spectrum),
          ", \"templates\": [", JoinStringsWithSeparator(tparts, ", "), "]",
          "}\n");
end;
