# Proyecto A Caso 2 - Urban Logistics (Bogotá)

## Project Context

**LogistiCo Urban Logistics Division** operates last-mile delivery services in Bogotá's metropolitan area. This case focuses on optimizing distribution with:

- **Multiple distribution centers** with limited inventory capacity
- **Urban delivery constraints** (traffic, vehicle type restrictions by zone)
- **Heterogeneous vehicle fleet** (small vans, medium vans, light trucks)
- **High delivery density** in congested urban environment

**Case Level**: Simplified project-specific (25% of implementation grade)

## Data Files

### 1. `clients.csv`

Urban delivery locations across Bogotá.

| Column         | Type    | Unit    | Description                                              |
| -------------- | ------- | ------- | -------------------------------------------------------- |
| ClientID       | Integer | -       | Numeric client identifier                                |
| StandardizedID | String  | -       | Standardized ID (C001, C002, ...) for verification files |
| LocationID     | Integer | -       | Unified location identifier                              |
| Latitude       | Float   | degrees | Geographic latitude in Bogotá                            |
| Longitude      | Float   | degrees | Geographic longitude in Bogotá                           |
| Demand         | Float   | kg      | Quantity of goods required                               |

### 2. `vehicles.csv`

Urban delivery fleet with varying capacities.

| Column         | Type    | Unit | Description                                              |
| -------------- | ------- | ---- | -------------------------------------------------------- |
| VehicleID      | Integer | -    | Numeric vehicle identifier                               |
| VehicleType    | String  | -    | Type of vehicle (small van, medium van, light truck)     |
| StandardizedID | String  | -    | Standardized ID (V001, V002, ...) for verification files |
| Capacity       | Float   | kg   | Maximum load capacity                                    |
| Range          | Float   | km   | Maximum travel distance on full tank                     |

**Note**: Vehicle types (small van, medium van, light truck) determine fuel efficiency - see `parameters_urban.csv`.

### 3. `depots.csv`

Multiple distribution centers with inventory limits.

| Column         | Type    | Unit    | Description                                              |
| -------------- | ------- | ------- | -------------------------------------------------------- |
| DepotID        | Integer | -       | Numeric depot identifier                                 |
| StandardizedID | String  | -       | Standardized ID (CD01, CD02, ...) for verification files |
| LocationID     | Integer | -       | Unified location identifier                              |
| Latitude       | Float   | degrees | Geographic latitude                                      |
| Longitude      | Float   | degrees | Geographic longitude                                     |
| **Capacity**   | Float   | kg      | **Maximum inventory available at this depot**            |

**KEY CONSTRAINT**: Each depot has limited inventory. Sum of initial loads for vehicles departing from a depot cannot exceed depot capacity.

## Cost Parameters

See `parameters_urban.csv` for all values. Key parameters from the unified objective function:

### Fixed and Variable Costs:

- **C_fixed**: 50,000 COP/vehicle (activation cost)
- **C_dist**: 2,500 COP/km (maintenance and wear)
- **C_time**: 7,600 COP/hour (urban driver wage)
- **fuel_price**: 16,300 COP/gallon

### Fuel Efficiency by Vehicle Type:

| Vehicle Type | Min Efficiency | Max Efficiency |
| ------------ | -------------- | -------------- |
| Small Van    | 35 km/gal      | 45 km/gal      |
| Medium Van   | 25 km/gal      | 35 km/gal      |
| Light Truck  | 22 km/gal      | 28 km/gal      |

### Objective Function:

```
min Z = Σ(C_fixed × y_v) + Σ(C_dist × d_v) + Σ(C_time × t_v) + C_fuel

Where:
- y_v = 1 if vehicle v is used, 0 otherwise
- d_v = total distance traveled by vehicle v (km)
- t_v = total operation time for vehicle v (hours)
- C_fuel = Σ(d_v / efficiency_v) × fuel_price
```

## Project-Specific Constraints

1. **Multiple Depots**: Vehicles can originate from any depot
2. **Depot Inventory**: Σ(initial loads from depot d) ≤ Capacity_d
3. **Vehicle-Depot Assignment**: Each vehicle departs from exactly one depot
4. **Capacity & Range**: Standard CVRP constraints apply
5. **Coverage**: All clients must be visited exactly once

## Verification File Format

Create `verificacion_caso2.csv`:

```csv
VehicleId,DepotId,InitialLoad,RouteSequence,ClientsServed,DemandsSatisfied,TotalDistance,TotalTime,FuelCost
V001,CD01,750,CD01-C005-C023-C017-CD01,3,215-320-215,28.4,67.2,98500
V002,CD02,680,CD02-C002-C009-C014-CD02,3,250-180-250,32.1,75.8,112300
```

**Key Points**:

- **DepotId**: Specify which depot each vehicle uses (CD01, CD02, CD03)
- Route must start and end at the SAME depot
- Use StandardizedID format for all IDs

## Deliverables for Caso 2

1. ✓ Correct implementation of multi-depot model (15%)
2. ✓ Feasible solution respecting depot capacities (7%)
3. ✓ Basic visualization and reporting (3%)
   - Map showing routes from each depot
   - Table of depot utilization (inventory used vs. available)

## Visualization Requirements

Create an interactive map using Folium showing:

- All distribution centers
- Client locations
- Routes color-coded by depot of origin
- Route sequence clearly marked

## Important Notes

- **Zone restrictions** by vehicle type may apply in Caso 3 (not in Caso 2)
- Focus on **feasibility** and correct constraint handling
- Depot capacity violations will result in infeasible solution
- Time calculations: Estimate using average urban speed (~25 km/h with stops)

## ID Standardization

**Always use StandardizedID in verification files**:

- Clients: C001, C002, C003, ...
- Vehicles: V001, V002, V003, ...
- Depots: CD01, CD02, CD03, ...

These correspond to the numeric IDs in your data files but follow the required format for grading.
