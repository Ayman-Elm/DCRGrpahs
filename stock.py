# Monthly investment growth with per-month display

monthly_contribution = 5000        # DKK per month
monthly_return_rate = 0.0066        # 1.5% per month
years = 30

months = years * 12
balance = 0.0

print(f"{'Month':>5} | {'Year':>4} | {'Balance at end of month':>25}")
print("-" * 45)

for month in range(1, months + 1):
    year = (month - 1) // 12 + 1  # which year we’re in (1–30)

    # First: existing balance grows by 1.5%
    balance *= (1 + monthly_return_rate)

    # Then: add this month's contribution
    balance += monthly_contribution

    # Print this month
    print(f"{month:5d} | {year:4d} | {balance:25,.2f}")

total_contributed = monthly_contribution * months
profit = balance - total_contributed

print("\nSummary after", years, "years")
print(f"  Total contributed: {total_contributed:,.2f}")
print(f"  Final balance:     {balance:,.2f}")
print(f"  Investment gain:   {profit:,.2f}")
