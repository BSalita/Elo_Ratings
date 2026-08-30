"""
Analyze the FFBridge handicap formula by comparing scratch vs handicapped scores.

This script fetches results from both scoring types and computes the formula
used to convert between them.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re


def fetch_results_page(session_code: str, club_code: str, classement_type: str) -> str:
    """
    Fetch a results page from bridgeinter.net.
    
    Args:
        session_code: Session code (e.g., 'lo260112')
        club_code: Club code (e.g., 'LEV')
        classement_type: 's' for scratch, 'h' for handicap
    
    Returns:
        HTML content of the page
    """
    url = f"http://www.bridgeinter.net/octopus_l/restotal.php?v_codeclub={club_code}&v_type_classement={classement_type}&v_codeseance={session_code}"
    print(f"Fetching: {url}")
    
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def parse_results_table(html: str) -> pd.DataFrame:
    """
    Parse the results table from the HTML.
    
    Returns DataFrame with columns: rank, position, players, percentage, iv_values
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Find all tables and look for the results table
    tables = soup.find_all('table')
    
    results = []
    seen_pairs = set()
    
    # The results are in a table with player links
    for table in tables:
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 2:
                # Look for player links (feuilleroute.php)
                links = row.find_all('a', href=lambda x: x and 'feuilleroute' in x)
                if len(links) >= 2:
                    # Extract player names
                    player1 = links[0].get_text(strip=True)
                    player2 = links[1].get_text(strip=True)
                    
                    pair_key = f"{player1}|{player2}"
                    
                    # Skip if we've already seen this pair
                    if pair_key in seen_pairs:
                        continue
                    seen_pairs.add(pair_key)
                    
                    # Get the text content of all cells
                    cell_texts = [cell.get_text(strip=True) for cell in cells]
                    row_text = row.get_text()
                    
                    # Extract percentage (look for pattern like "62.41%")
                    pct_match = re.search(r'(\d+\.\d+)%', row_text)
                    if pct_match:
                        percentage = float(pct_match.group(1))
                        
                        # Try to parse the cell values
                        # Format appears to be: Rank, IV1(?), IV2(?), Position, Players, Percentage, Points...
                        rank = None
                        position = None
                        iv1 = None
                        iv2 = None
                        
                        # Extract position (NS or EO)
                        pos_match = re.search(r'\b(NS|EO)\b', row_text)
                        position = pos_match.group(1) if pos_match else None
                        
                        # Parse numeric values from cells
                        numeric_cells = []
                        for i, txt in enumerate(cell_texts):
                            # Try to extract numbers
                            nums = re.findall(r'\d+', txt)
                            if nums and 'feuilleroute' not in str(cells[i]):
                                for n in nums:
                                    numeric_cells.append((i, int(n)))
                        
                        # First number is usually rank
                        if numeric_cells:
                            rank = numeric_cells[0][1] if numeric_cells[0][1] <= 50 else None
                        
                        # After percentage, look for numbers that could be IV values
                        # In FFBridge tables, IV values are often in columns after percentage
                        after_pct = row_text[row_text.find('%')+1:]
                        after_pct_numbers = re.findall(r'\d+', after_pct)
                        
                        # Try to find IV values - they're typically < 40 each (sum < 80)
                        # and appear after the percentage
                        potential_ivs = []
                        for n in after_pct_numbers:
                            val = int(n)
                            if 0 <= val <= 300:  # Reasonable range for IV or points
                                potential_ivs.append(val)
                        
                        # Try to extract IV-like values from cells
                        # Format appears to be: Rank, Col2, Col3, Position, Player1, Player2, Pct%, ...
                        col2 = None
                        col3 = None
                        if len(cell_texts) >= 4:
                            try:
                                # First cell is rank, second and third might be IV-related
                                if cell_texts[1].isdigit():
                                    col2 = int(cell_texts[1])
                                if cell_texts[2].isdigit():
                                    col3 = int(cell_texts[2])
                            except (ValueError, IndexError):
                                pass
                        
                        results.append({
                            'rank': rank,
                            'position': position,
                            'player1': player1,
                            'player2': player2,
                            'pair_key': pair_key,
                            'percentage': percentage,
                            'col2': col2,
                            'col3': col3,
                            'cells': cell_texts,
                            'after_pct_nums': potential_ivs
                        })
    
    return pd.DataFrame(results)


def analyze_handicap_formula(scratch_df: pd.DataFrame, handicap_df: pd.DataFrame) -> Dict:
    """
    Analyze the difference between scratch and handicap scores to derive the formula.
    """
    # Deduplicate DataFrames - keep first occurrence of each pair
    scratch_df = scratch_df.drop_duplicates(subset=['pair_key'], keep='first').copy()
    handicap_df = handicap_df.drop_duplicates(subset=['pair_key'], keep='first').copy()
    
    # Merge on pair_key
    merged = scratch_df.merge(
        handicap_df,
        on='pair_key',
        suffixes=('_scratch', '_handicap')
    )
    
    if merged.empty:
        # Try matching on sorted player names (in case order differs)
        scratch_df['pair_key_sorted'] = scratch_df.apply(
            lambda r: '|'.join(sorted([r['player1'], r['player2']])), axis=1
        )
        handicap_df['pair_key_sorted'] = handicap_df.apply(
            lambda r: '|'.join(sorted([r['player1'], r['player2']])), axis=1
        )
        merged = scratch_df.merge(
            handicap_df,
            on='pair_key_sorted',
            suffixes=('_scratch', '_handicap')
        )
    
    if merged.empty:
        return {'error': 'Could not match pairs between scratch and handicap results'}
    
    # Calculate the bonus (difference)
    merged['bonus'] = merged['percentage_handicap'] - merged['percentage_scratch']
    
    # Analyze the bonus distribution
    analysis = {
        'num_pairs': len(merged),
        'mean_bonus': merged['bonus'].mean(),
        'std_bonus': merged['bonus'].std(),
        'min_bonus': merged['bonus'].min(),
        'max_bonus': merged['bonus'].max(),
        'pairs': merged[['player1_scratch', 'player2_scratch', 
                         'percentage_scratch', 'percentage_handicap', 'bonus']].to_dict('records'),
        'raw_data': merged
    }
    
    return analysis


def fetch_national_ranking(session_code: str, classement_type: str) -> Optional[str]:
    """
    Try to fetch national ranking (without club code).
    """
    # Try different URL patterns for national ranking
    url_patterns = [
        f"http://www.bridgeinter.net/octopus_l/restotal.php?v_type_classement={classement_type}&v_codeseance={session_code}",
        f"http://www.bridgeinter.net/octopus_l/restotal.php?v_codeclub=TOUS&v_type_classement={classement_type}&v_codeseance={session_code}",
        f"http://www.bridgeinter.net/octopus_l/resgen.php?v_type_classement={classement_type}&v_codeseance={session_code}",
    ]
    
    for url in url_patterns:
        try:
            print(f"  Trying: {url}")
            response = requests.get(url, timeout=10)
            if response.status_code == 200 and len(response.text) > 1000:
                # Check if it contains results data
                if 'feuilleroute' in response.text:
                    print(f"  -> Found national ranking!")
                    return response.text
        except Exception as e:
            print(f"  -> Error: {e}")
    
    return None


def main():
    """Main function to analyze handicap formula."""
    
    # Session to analyze
    session_code = "lo260112"  # January 12, 2026 Levallois session
    club_code = "LEV"
    
    print("=" * 70)
    print("FFBridge Handicap Formula Analysis")
    print("=" * 70)
    print(f"\nSession: {session_code}")
    print(f"Club: {club_code}")
    print()
    
    # First, try to find national rankings to test hypothesis
    print("Testing hypothesis: Is handicap score national while scratch is local?")
    print("-" * 70)
    print("\nLooking for national rankings...")
    national_scratch = fetch_national_ranking(session_code, 's')
    national_handicap = fetch_national_ranking(session_code, 'h')
    
    if national_scratch:
        national_scratch_df = parse_results_table(national_scratch)
        print(f"  National scratch found: {len(national_scratch_df)} pairs")
    else:
        print("  No national scratch ranking found")
        national_scratch_df = None
        
    if national_handicap:
        national_handicap_df = parse_results_table(national_handicap)
        print(f"  National handicap found: {len(national_handicap_df)} pairs")
    else:
        print("  No national handicap ranking found")
        national_handicap_df = None
    
    # Store national dataframes for later comparison
    nat_scratch_df = national_scratch_df
    nat_handicap_df = national_handicap_df
    
    # Compare national scratch vs national handicap to isolate IV formula
    if national_scratch_df is not None and national_handicap_df is not None:
        print("\n" + "=" * 70)
        print("NATIONAL SCRATCH vs NATIONAL HANDICAP (IV Formula Isolation)")
        print("=" * 70)
        
        nat_analysis = analyze_handicap_formula(national_scratch_df, national_handicap_df)
        if 'error' not in nat_analysis:
            print(f"\nMatched {nat_analysis['num_pairs']} pairs at national level")
            print(f"\nNational Bonus Statistics (Handicap% - Scratch%):")
            print(f"  Mean:   {nat_analysis['mean_bonus']:+.2f}%")
            print(f"  StdDev: {nat_analysis['std_bonus']:.2f}%")
            print(f"  Min:    {nat_analysis['min_bonus']:+.2f}%")
            print(f"  Max:    {nat_analysis['max_bonus']:+.2f}%")
            
            print("\nTop 10 pairs by HIGHEST bonus (weakest players):")
            print(f"{'Players':<45} {'Scratch':>8} {'Hcap':>8} {'Bonus':>8}")
            print("-" * 70)
            for pair in sorted(nat_analysis['pairs'], key=lambda x: -x['bonus'])[:10]:
                players = f"{pair['player1_scratch']} / {pair['player2_scratch']}"
                if len(players) > 44:
                    players = players[:41] + "..."
                print(f"{players:<45} {pair['percentage_scratch']:>7.2f}% {pair['percentage_handicap']:>7.2f}% {pair['bonus']:>+7.2f}%")
            
            print("\nTop 10 pairs by LOWEST bonus (strongest players):")
            print(f"{'Players':<45} {'Scratch':>8} {'Hcap':>8} {'Bonus':>8}")
            print("-" * 70)
            for pair in sorted(nat_analysis['pairs'], key=lambda x: x['bonus'])[:10]:
                players = f"{pair['player1_scratch']} / {pair['player2_scratch']}"
                if len(players) > 44:
                    players = players[:41] + "..."
                print(f"{players:<45} {pair['percentage_scratch']:>7.2f}% {pair['percentage_handicap']:>7.2f}% {pair['bonus']:>+7.2f}%")
            
            # Unique bonus values
            unique_bonuses = sorted(set(round(p['bonus']) for p in nat_analysis['pairs']))
            print(f"\nUnique bonus values (national): {unique_bonuses}")
    
    print()
    
    # Fetch club-level pages
    try:
        scratch_html = fetch_results_page(session_code, club_code, 's')
        handicap_html = fetch_results_page(session_code, club_code, 'h')
    except Exception as e:
        print(f"Error fetching pages: {e}")
        return
    
    # Parse results
    print("\nParsing scratch results...")
    scratch_df = parse_results_table(scratch_html)
    print(f"  Found {len(scratch_df)} pairs")
    
    print("\nParsing handicap results...")
    handicap_df = parse_results_table(handicap_html)
    print(f"  Found {len(handicap_df)} pairs")
    
    if scratch_df.empty or handicap_df.empty:
        print("ERROR: Could not parse results tables")
        return
    
    # Display scratch results with raw numbers for IV analysis
    print("\n" + "=" * 70)
    print("SCRATCH (Unhandicapped) Results")
    print("=" * 70)
    display_cols = ['rank', 'position', 'player1', 'player2', 'percentage']
    if 'after_pct_nums' in scratch_df.columns:
        display_cols.append('after_pct_nums')
    print(scratch_df[display_cols].head(15).to_string(index=False))
    
    # Show raw cells for first few rows to understand structure
    print("\nRaw cell values for first 3 rows:")
    for i, row in scratch_df.head(3).iterrows():
        print(f"  Row {i}: {row.get('cells', [])}")
    
    # Display handicap results  
    print("\n" + "=" * 70)
    print("HANDICAP Results")
    print("=" * 70)
    display_cols = ['rank', 'position', 'player1', 'player2', 'percentage']
    if 'after_pct_nums' in handicap_df.columns:
        display_cols.append('after_pct_nums')
    print(handicap_df[display_cols].head(15).to_string(index=False))
    
    # Analyze the formula
    print("\n" + "=" * 70)
    print("HANDICAP BONUS ANALYSIS")
    print("=" * 70)
    
    analysis = analyze_handicap_formula(scratch_df, handicap_df)
    
    if 'error' in analysis:
        print(f"ERROR: {analysis['error']}")
        return
    
    print(f"\nMatched {analysis['num_pairs']} pairs between scratch and handicap")
    print(f"\nBonus Statistics (Handicap% - Scratch%):")
    print(f"  Mean:   {analysis['mean_bonus']:+.2f}%")
    print(f"  StdDev: {analysis['std_bonus']:.2f}%")
    print(f"  Min:    {analysis['min_bonus']:+.2f}%")
    print(f"  Max:    {analysis['max_bonus']:+.2f}%")
    
    print("\n" + "-" * 70)
    print("Per-Pair Breakdown:")
    print("-" * 70)
    print(f"{'Players':<45} {'Scratch':>8} {'Hcap':>8} {'Bonus':>8}")
    print("-" * 70)
    
    for pair in sorted(analysis['pairs'], key=lambda x: -x['bonus']):
        players = f"{pair['player1_scratch']} / {pair['player2_scratch']}"
        if len(players) > 44:
            players = players[:41] + "..."
        print(f"{players:<45} {pair['percentage_scratch']:>7.2f}% {pair['percentage_handicap']:>7.2f}% {pair['bonus']:>+7.2f}%")
    
    # Try to infer the formula
    print("\n" + "=" * 70)
    print("FORMULA INFERENCE")
    print("=" * 70)
    
    pairs_data = analysis['pairs']
    
    # Check if bonus correlates with scratch percentage
    scratch_pcts = [p['percentage_scratch'] for p in pairs_data]
    bonuses = [p['bonus'] for p in pairs_data]
    
    if len(scratch_pcts) > 2:
        correlation = np.corrcoef(scratch_pcts, bonuses)[0, 1]
        print(f"\nCorrelation between Scratch% and Bonus: {correlation:.3f}")
        
        # Linear regression: bonus = a * scratch_pct + b
        A = np.vstack([scratch_pcts, np.ones(len(scratch_pcts))]).T
        slope, intercept = np.linalg.lstsq(A, bonuses, rcond=None)[0]
        
        print(f"\nLinear fit: Bonus = {slope:.4f} x Scratch% + {intercept:.2f}")
        print(f"Or: Handicap% = {1 + slope:.4f} x Scratch% + {intercept:.2f}")
        
        # Predict and check residuals
        predicted = [slope * s + intercept for s in scratch_pcts]
        residuals = [b - p for b, p in zip(bonuses, predicted)]
        rmse = np.sqrt(np.mean([r**2 for r in residuals]))
        print(f"RMSE of linear fit: {rmse:.3f}%")
        
        # Check if bonus is approximately constant (simple additive)
        if analysis['std_bonus'] < 1.0:
            print(f"\n-> Bonus appears roughly constant: Handicap% = Scratch% + {analysis['mean_bonus']:.2f}%")
        elif abs(correlation) > 0.5:
            print(f"\n-> Bonus correlates with Scratch%: stronger players get smaller bonus")
        else:
            print(f"\n-> Bonus varies but doesn't strongly correlate with Scratch%")
            print("   This confirms IV (player strength rating) determines the bonus")
            
            # Analyze unique bonus values
            unique_bonuses = sorted(set(bonuses))
            print(f"\n   Unique bonus values observed: {unique_bonuses}")
            print(f"   All bonuses are integers: {all(b == int(b) for b in bonuses)}")
            
            print("\n" + "=" * 70)
            print("CONCLUSION: FFBridge Handicap Formula")
            print("=" * 70)
            print("\n   Handicap% = Scratch% + Bonus(IV)")
            print("\n   Where Bonus is determined by the pair's combined IV (Indice de Valeur)")
            print("   relative to the field average IV.")
            print("\n   Lower IV pairs (weaker players) receive larger bonuses.")
            print("   Higher IV pairs (stronger players) receive smaller bonuses.")
            
            # Try to analyze IV-Bonus relationship from raw data
            if 'raw_data' in analysis:
                raw = analysis['raw_data']
                if 'after_pct_nums_scratch' in raw.columns:
                    print("\n" + "=" * 70)
                    print("IV VALUE EXTRACTION ATTEMPT")
                    print("=" * 70)
                    
                    # Show full data for analysis
                    print("\nFull data with after-percentage numbers:")
                    for _, row in raw.iterrows():
                        players = f"{row['player1_scratch']}/{row['player2_scratch']}"[:35]
                        bonus = row['bonus']
                        after_nums = row.get('after_pct_nums_scratch', [])
                        print(f"  {players:<35} Bonus: {bonus:+5.0f}%  Nums: {after_nums}")
                    
                    # Try to find correlation between after_pct_nums and bonus
                    print("\nLooking for IV->Bonus relationship...")


if __name__ == "__main__":
    main()
