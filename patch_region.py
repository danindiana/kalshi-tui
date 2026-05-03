        row  = np.concatenate([base_feats, [dist, mins]]).reshape(1, -1)
        clf_prob = float(clf.predict_proba(row)[0][1])   # P(price > strike)

        yes_price = mk["yes_price"]
        no_price  = mk.get("no_price", 1.0 - yes_price)
        if clf_prob > 0.5:
            side         = "YES"
            model_prob   = clf_prob
            market_price = yes_price
            order_price  = yes_price
        else:
            side         = "NO"
            model_prob   = 1.0 - clf_prob
            market_price = 1.0 - yes_price
            order_price  = no_price

        edge = model_prob - market_price
        if edge < MIN_EDGE:
            continue

        # Gate the *actual* order price (no_price may diverge from 1-yes_price
        # when the book is crossed or one side has no offers — pre-fix, these
        # fell through to Kalshi and returned 400 errors).
        contract_cents = round(order_price * 100)
        if contract_cents < MIN_CONTRACT_CENTS:
            print(f"  [SKIP] Strike ${strike:,.0f}: {contract_cents}¢ < {MIN_CONTRACT_CENTS:.0f}¢ min")
            continue
        if contract_cents >= MAX_CONTRACT_CENTS:
            print(f"  [SKIP] Strike ${strike:,.0f}: {contract_cents}¢ >= {MAX_CONTRACT_CENTS:.0f}¢ max (Kalshi rejects at-limit contracts)")
            continue
        if market_price > 0 and model_prob / market_price > MAX_PROB_RATIO:
