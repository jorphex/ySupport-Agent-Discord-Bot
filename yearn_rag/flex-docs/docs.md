---
title: "Flex - Docs"
slug: "/docs"
source_url: "https://flexmeow.com/docs"
---

# Flex - Docs

## Overview

### What is Flex

Flex is a fixed-rate money market where borrowers choose their own interest rate.

### Who is Flex for

Borrowers

Borrowers who want to choose a fixed interest rate instead of accepting a protocol-defined rate.

Lenders

Lenders who want to earn interest on their tokens while maintaining liquidity through redemptions.

### Who is Flex not for

Borrowers

Borrowers who are unwilling to have their position redeemed by someone else.

Lenders

Lenders who are unwilling to pay swap fees to exit.

### How is Flex different from Aave

Aave uses variable interest rates defined by utilization curves.

Flex uses borrower-defined fixed rates enforced by the market.

Aave maintains liquidity by adjusting rates.

Flex maintains liquidity through redemptions.

### How is Flex different from Liquity

Flex borrows Liquity's V2 ideas of fixed interest rates and liquidity managed through redemptions.

Liquity is a stablecoin issuer, while Flex is a general-purpose money market connecting borrowers and lenders.

## Glossary

### Trove

A Trove is a borrower's loan together with its collateral. Each address may own one or more Troves.

### Lender Vault

A ERC4626 Yearn tokenized strategy vault where lenders deposit borrow tokens. Each lender owns shares of the vault, whose value increases over time as interest and fees are earned.

### Collateral

Tokens locked in a Trove to secure a loan (e.g. wstETH).

### Debt

Tokens owed by a Trove (e.g. USDC).

### Redemption

The process by which a borrower's debt is reduced by selling part or all of their collateral.

### Redeemer

A user who redeems someone else.

### Liquidation

The forced closure of a Trove when its collateral value falls below the minimum required level.

### Dutch Auction

A type of auction where the price starts high and gradually decreases until a buyer is found.

### Average Interest Rate

The average interest rate paid by all active borrowers in the system.

### Upfront Fee

A fee paid when opening or increasing a loan.

### Premature Rate Adjustment Fee

A fee paid when changing a Trove's interest rate too frequently.

### Minimum Debt

The minimum amount of debt required for a Trove to remain open.

### Zombie Trove

A Trove whose debt has been reduced below the minimum and can no longer be adjusted, only closed.

## Flex for Borrowers

### Borrowing on Flex

When borrowing on Flex, borrowers open a Trove by (1) locking collateral (2) choosing how much to borrow and (3) choosing a fixed interest rate

Once the Trove is opened, the interest rate does not change unless the borrower chooses to update it.

### What happens after you borrow

After a Trove is opened, it may change over time.

At any point, another user may reduce your debt. When this happens, a corresponding amount of your collateral is sold.

This process is called a redemption and is how liquidity is maintained on Flex.

### What you control

Borrowers control (1) the amount they borrow (2) the interest rate they pay (3) how much collateral they lock and (4) when they repay or close their Trove.

### What you do not control

Borrowers do not control (1) when redemptions occur (2) how much of their debt is reduced through a redemption and (3) the price at which collateral is sold during a redemption.

### How interest rates affect redemptions

Borrowers can reduce the likelihood of being affected by redemptions by choosing a higher interest rate.

When redemptions occur, they start with Troves offering the lowest interest rates and move upward.

Choosing a higher rate does not prevent redemptions, but it makes them less likely.

### Keeping your Trove healthy

Borrowers are responsible for ensuring their Trove remains sufficiently collateralized.

If a Trove's collateral value falls below the minimum required level, it may be liquidated and closed.

### Fees

Borrowers may pay fees when (1) opening or increasing a loan and (2) adjusting their interest rate.

## Flex for Lenders

### Lending on Flex

Lenders provide tokens to a Flex market in order to earn interest from borrowers.

Interest earned by lenders comes from the fixed interest rates chosen by borrowers.

### The Lender Vault

Lenders deposit their borrow tokens into a Lender Vault. The Lender Vault is a ERC4626 Yearn tokenized strategy vault that represents a lender's share of the market.

As interest and fees are earned, the value of each vault share increases over time.

### Liquidity on Flex

Lenders can exit at any time.

If there is not enough idle liquidity available, exiting lenders reduce existing Troves' debt by selling a corresponding portion of the borrowers' collateral.

This process is called a redemption.

### What affects your returns

Lenders earn from three sources, (1) interest paid by borrowers, (2) upfront fees paid by borrowers and (3) any surplus from liquidated collateral.

### Costs of exiting

Exiting a position may involve costs.

If sufficient idle liquidity is available, lenders can exit without cost.

If not, part of the exit is executed through redemptions, which may involve market-based costs depending on collateral prices and execution conditions at the time.

## Fees

Flex uses fees to align incentives between borrowers and lenders and to keep the system fair.

### Fees for Borrowers

Borrowers may pay the following fees.

Interest Rate

Borrowers pay interest at the fixed rate they choose.

Upfront Fee

When opening or increasing a loan, borrowers pay an upfront fee. The upfront fee equals one week of the market's average interest rate.

Its purpose is to discourage borrowers from choosing unrealistically low interest rates that would immediately be redeemed.

Premature Rate Adjustment Fee

Borrowers who change their interest rate too frequently pay a fee.

This fee exists to prevent borrowers from temporarily increasing their rate to avoid redemptions and then immediately lowering it again.

### Fees for Lenders

Performance Fee

Lenders pay a 10% performance fee to the protocol.

## Flex by Example

A tale of Alice, Bob, and Yossi

1. Alice lends 500 USDC into the wstETH/USDC market

2. Bob opens a Trove

- Deposits $1,000 of wstETH

- Borrows 500 USDC

- Chooses a 4% interest rate

Now there are two possible scenarios.

#### Scenario 1 - Alice exits

3. Alice exits the market

4. There is no idle liquidity, so Bob's Trove is redeemed

5. 500 USDC worth of Bob's wstETH is sold for 500 USDC, which is used to repay Bob's debt and sent to Alice

Result:

- Alice exits with 500 USDC

- Bob is left with 0 debt, 500 USDC, and $500 of wstETH

#### Scenario 2 — Yossi borrows instead

3. Yossi opens a Trove with a 5% interest rate (higher than Bob's 4%)

4. There is no idle liquidity, so Bob's Trove is redeemed

5. 500 USDC worth of Bob's wstETH is sold for 500 USDC, which is used to repay Bob's debt and sent to Yossi

Result:

- Yossi now has a Trove with 500 USDC debt

- Bob is left with 0 debt, 500 USDC, and $500 of wstETH

Note: In practice, the redeemer receives the borrow tokens minus any market and swap fees incurred when selling the collateral.

## How Collateral Is Sold

When a redemption occurs, collateral taken from a Trove must be sold for the borrow token.

Flex sells collateral using Dutch auctions.

### What is a Dutch auction

In a Dutch auction:

- The price starts high

- The price gradually decreases over time

- The first buyer willing to accept the price completes the trade

### How this is used in Flex

During a redemption:

- Collateral is placed into a Dutch auction

- The auction runs until a buyer is found

- The proceeds are paid to the redeemer

This process happens asynchronously and does not rely on predefined swap paths.

### What this means for users

- Borrowers do not control the price at which their collateral is sold during a redemption

- Lenders and redeemers may receive more or less than the nominal amount, depending on market conditions

This is why redemption proceeds can differ from the headline amounts shown in examples.

### Why Flex uses Dutch auctions

Two reasons:

- Anyone can compete to fill an auction, so Flex doesn't need to integrate with DEXes or actively manage routes.

- The starting price is set above market, which prevents instant arbitrage when Flex's price drifts from the market's.

## Taking and Re-kicking Auctions

Every redemption creates a Dutch auction. The Auctions page lists each market's live and recent auctions and lets anyone act on them.

### Taking an auction

While an auction is live, its price decreases over time.

Once the current price reaches or falls below the market price, taking the auction becomes profitable. A Take button enables on the auction page and anyone can click it to settle the auction.

If you don't, bots typically pick up auctions automatically as soon as they become profitable.

### Re-kicking an auction

If an auction reaches its minimum price without being filled (e.g. the amount is too small to be worth taking, or market conditions move against it), it sits unsettled.

At that point a Re-Kick button enables on the auction page. Anyone can click it to restart the auction with a fresh starting price.

## Liquidations and Bad Debt

If a Trove's collateral value falls below the minimum required level, anyone can liquidate it. Liquidators repay part (or all) of the debt and receive the matching collateral plus a bonus.

### How much can a liquidator repay

Liquidators choose how much debt to repay, with two limits:

- They can only repay enough to bring the Trove back to a safe collateral ratio. This avoids liquidating more than necessary.

- They can't leave dust. If the remaining debt would fall below a threshold, the Trove must be fully liquidated.

### Liquidation bonus

The bonus liquidators receive scales with how unhealthy the Trove is. Riskier positions offer higher bonuses, incentivizing liquidators to clear the most unhealthy Troves first.

### Bad debt

If a Trove's collateral is worth less than its debt, liquidators still have an incentive to act.

The liquidator gets all the collateral and the entire remaining debt is cleared from the system. The shortfall is socialized as a loss to lenders atomically.
