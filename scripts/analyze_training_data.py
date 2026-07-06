#!/usr/bin/env python3
"""
Analyze class distribution in multi-county training data.
Use to diagnose class imbalance issues across counties.
"""

import geopandas as gpd
import sys
from pathlib import Path


def main():
    # Paths
    labels_path = Path(__file__).parent.parent / "data" / "multi_county_labels.gpkg"
    
    if not labels_path.exists():
        print(f"ERROR: {labels_path} not found")
        sys.exit(1)
    
    print(f"Loading {labels_path}...\n")
    gdf = gpd.read_file(labels_path)
    
    print("=" * 70)
    print("=== OVERALL CLASS DISTRIBUTION ===")
    print("=" * 70)
    print(f"\nTotal features: {len(gdf)}")
    print(f"Unique classes: {sorted(gdf['Classname'].unique())}")
    print(f"\nClass counts:")
    for cls, count in gdf['Classname'].value_counts().items():
        pct = count / len(gdf) * 100
        print(f"  {cls:20s}: {count:5d} ({pct:5.1f}%)")
    
    print("\n" + "=" * 70)
    print("=== PER-COUNTY CLASS DISTRIBUTION ===")
    print("=" * 70)
    
    county_stats = []
    
    for county in sorted(gdf['county'].unique()):
        county_gdf = gdf[gdf['county'] == county]
        total = len(county_gdf)
        
        tile_outlet_pct = (county_gdf['Classname'] == 'Tile_Outlet').sum() / total * 100
        bank_erosion_pct = (county_gdf['Classname'] == 'Bank_Erosion').sum() / total * 100
        
        county_stats.append({
            'county': county,
            'total': total,
            'tile_outlet_pct': tile_outlet_pct,
            'bank_erosion_pct': bank_erosion_pct
        })
        
        print(f"\n{county}: {total} features")
        for cls, count in county_gdf['Classname'].value_counts().items():
            pct = count / total * 100
            print(f"  {cls:20s}: {count:5d} ({pct:5.1f}%)")
    
    print("\n" + "=" * 70)
    print("=== SUSPICIOUS COUNTIES (Tile_Outlet > 50% OR Bank_Erosion < 20%) ===")
    print("=" * 70)
    
    suspicious = [s for s in county_stats if s['tile_outlet_pct'] > 50 or s['bank_erosion_pct'] < 20]
    
    if suspicious:
        print(f"\nFound {len(suspicious)} suspicious county(ies):\n")
        for s in sorted(suspicious, key=lambda x: x['tile_outlet_pct'], reverse=True):
            print(f"  {s['county']:15s}: {s['tile_outlet_pct']:5.1f}% Tile_Outlet, {s['bank_erosion_pct']:5.1f}% Bank_Erosion")
    else:
        print("\nNo suspicious counties found.")


if __name__ == '__main__':
    main()
