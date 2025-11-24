


import os
import math
from typing import Dict, List, Tuple

import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, NonNegativeReals, Binary,
    Expression, Objective, Constraint, SolverFactory, value, minimize
)


# -----------------------------
# Utilidades de carga de datos
# -----------------------------

def load_data(data_dir: str = "data"):
    """
    Lee los archivos CSV requeridos desde data_dir y devuelve
    dataframes y estructuras auxiliares.
    """
    clients_path = os.path.join(data_dir, "clients.csv")
    depots_path = os.path.join(data_dir, "depots.csv")
    vehicles_path = os.path.join(data_dir, "vehicles.csv")
    params_path = os.path.join(data_dir, "parameters_urban.csv")

    if not os.path.exists(clients_path):
        raise FileNotFoundError(f"No se encontró {clients_path}")
    if not os.path.exists(depots_path):
        raise FileNotFoundError(f"No se encontró {depots_path}")
    if not os.path.exists(vehicles_path):
        raise FileNotFoundError(f"No se encontró {vehicles_path}")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"No se encontró {params_path}")

    clients_df = pd.read_csv(clients_path)
    depots_df = pd.read_csv(depots_path)
    vehicles_df = pd.read_csv(vehicles_path)
    params_df = pd.read_csv(params_path)

    # Arreglo de vehicles.csv si viene con VehicleType y StandardizedID intercambiados
    # Según README:
    #   VehicleType    = "small van", "medium van", "light truck"
    #   StandardizedID = "V001", "V002", ...
    # En el archivo entregado, VehicleType viene como "V001" y StandardizedID como "small van"
    if vehicles_df["VehicleType"].astype(str).str.startswith("V").all():
        tmp = vehicles_df["VehicleType"].copy()
        vehicles_df["VehicleType"] = vehicles_df["StandardizedID"]
        vehicles_df["StandardizedID"] = tmp

    # Validaciones mínimas
    if clients_df["Demand"].lt(0).any():
        raise ValueError("Hay demandas negativas en clients.csv")

    return {
        "clients_df": clients_df,
        "depots_df": depots_df,
        "vehicles_df": vehicles_df,
        "params_df": params_df,
    }


# -----------------------------
# Distancias y coordenadas
# -----------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia Haversine en km entre dos puntos (lat, lon) en grados."""
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2.0) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))


def build_coordinates(clients_df: pd.DataFrame, depots_df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    """
    Construye un diccionario id_estandarizado -> (lat, lon)
    para clientes y depósitos.
    """
    coords: Dict[str, Tuple[float, float]] = {}

    for _, row in clients_df.iterrows():
        coords[str(row["StandardizedID"])] = (float(row["Latitude"]), float(row["Longitude"]))

    for _, row in depots_df.iterrows():
        coords[str(row["StandardizedID"])] = (float(row["Latitude"]), float(row["Longitude"]))

    return coords


def build_distance_matrix(nodes: List[str], coords: Dict[str, Tuple[float, float]]) -> Dict[Tuple[str, str], float]:
    """
    Construye un diccionario (i,j) -> distancia_ij en km para todos los pares de nodos.
    """
    dist: Dict[Tuple[str, str], float] = {}
    for i in nodes:
        lat_i, lon_i = coords[i]
        for j in nodes:
            if i == j:
                dist[(i, j)] = 0.0
            else:
                lat_j, lon_j = coords[j]
                dist[(i, j)] = haversine_km(lat_i, lon_i, lat_j, lon_j)
    return dist


# -----------------------------
# Parámetros de costos
# -----------------------------

def build_cost_parameters(params_df: pd.DataFrame, vehicles_df: pd.DataFrame):
    """
    A partir de parameters_urban.csv y vehicles_df construye:
    - C_fixed, C_dist, C_time, fuel_price
    - eficiencia km/gal por tipo y por vehículo
    - costo de combustible por km por vehículo
    """
    p_series = params_df.set_index("Parameter")["Value"]

    C_fixed = float(p_series["C_fixed"])
    C_dist = float(p_series["C_dist"])
    C_time = float(p_series["C_time"])
    fuel_price = float(p_series["fuel_price"])

    eff_small = 0.5 * (p_series["fuel_efficiency_van_small_min"] + p_series["fuel_efficiency_van_small_max"])
    eff_medium = 0.5 * (p_series["fuel_efficiency_van_medium_min"] + p_series["fuel_efficiency_van_medium_max"])
    eff_truck = 0.5 * (p_series["fuel_efficiency_truck_light_min"] + p_series["fuel_efficiency_truck_light_max"])

    eff_type = {
        "small van": float(eff_small),
        "medium van": float(eff_medium),
        "light truck": float(eff_truck),
    }

    eff_v: Dict[str, float] = {}
    fuel_cost_per_km: Dict[str, float] = {}
    for _, row in vehicles_df.iterrows():
        v_id = str(row["StandardizedID"])
        v_type = str(row["VehicleType"])
        if v_type not in eff_type:
            raise ValueError(f"Tipo de vehículo desconocido para eficiencia: {v_type}")
        eff = eff_type[v_type]
        eff_v[v_id] = eff
        fuel_cost_per_km[v_id] = float(fuel_price) / eff

    return C_fixed, C_dist, C_time, fuel_price, eff_v, fuel_cost_per_km


# -----------------------------
# Construcción del modelo Pyomo
# -----------------------------

def build_model(data: dict, speed_kmph: float = 25.0) -> ConcreteModel:
    """
    Construye el modelo de Pyomo para el Caso 2, Proyecto A.
    """
    clients_df = data["clients_df"]
    depots_df = data["depots_df"]
    vehicles_df = data["vehicles_df"]
    params_df = data["params_df"]

    client_ids = [str(cid) for cid in clients_df["StandardizedID"].tolist()]
    depot_ids = [str(did) for did in depots_df["StandardizedID"].tolist()]
    vehicle_ids = [str(vid) for vid in vehicles_df["StandardizedID"].tolist()]

    # Demandas por cliente
    demand = {str(row["StandardizedID"]): float(row["Demand"]) for _, row in clients_df.iterrows()}

    # Capacidad de vehículos y rangos
    capacity_v = {str(row["StandardizedID"]): float(row["Capacity"]) for _, row in vehicles_df.iterrows()}
    range_v = {str(row["StandardizedID"]): float(row["Range"]) for _, row in vehicles_df.iterrows()}

    # Capacidad de depósitos
    cap_depot = {str(row["StandardizedID"]): float(row["Capacity"]) for _, row in depots_df.iterrows()}

    # Coordenadas y distancias
    coords = build_coordinates(clients_df, depots_df)
    nodes = client_ids + depot_ids
    distances = build_distance_matrix(nodes, coords)

    # Costos y eficiencia de combustible
    C_fixed, C_dist, C_time, fuel_price, eff_v, fuel_cost_per_km = build_cost_parameters(params_df, vehicles_df)

    # Construcción del modelo
    m = ConcreteModel(name="ProyectoA_Caso2")

    # Guardar parámetros globales en el objeto modelo para uso posterior
    m.C_fixed = C_fixed
    m.C_dist = C_dist
    m.C_time = C_time
    m.fuel_price = fuel_price

    # Conjuntos
    m.V = Set(initialize=vehicle_ids)   # vehículos
    m.I = Set(initialize=client_ids)    # clientes
    m.D = Set(initialize=depot_ids)     # depósitos
    m.N = Set(initialize=nodes)         # todos los nodos

    # Parámetros
    m.q = Param(m.I, initialize=demand, within=NonNegativeReals)
    m.Q = Param(m.V, initialize=capacity_v, within=NonNegativeReals)
    m.R = Param(m.V, initialize=range_v, within=NonNegativeReals)
    m.CapDepot = Param(m.D, initialize=cap_depot, within=NonNegativeReals)
    m.c = Param(m.N, m.N, initialize=distances, within=NonNegativeReals)

    m.fuel_cost_per_km = Param(m.V, initialize=fuel_cost_per_km, within=NonNegativeReals)
    m.speed_kmph = Param(initialize=float(speed_kmph))

    # Variables
    m.x = Var(m.V, m.N, m.N, within=Binary)           # arco i->j usado por v
    m.y = Var(m.V, within=Binary)                     # vehículo v usado
    m.z = Var(m.V, m.D, within=Binary)                # vehículo v asignado a depósito d
    m.u = Var(m.V, m.N, within=NonNegativeReals)      # carga restante en nodo n
    m.Init = Var(m.V, m.D, within=NonNegativeReals)   # carga inicial de v en depósito d

    # Fijar la diagonal y arcos entre depósitos a cero
    for v in m.V:
        for i in m.N:
            m.x[v, i, i].fix(0)
        for d1 in m.D:
            for d2 in m.D:
                m.x[v, d1, d2].fix(0)

    # 1. Cada cliente se visita exactamente una vez
    def visit_once_rule(m, i):
        return sum(m.x[v, i, j] for v in m.V for j in m.N if j != i) == 1
    m.VisitOnce = Constraint(m.I, rule=visit_once_rule)

    # 2. Conservación de flujo en clientes
    def flow_client_rule(m, v, k):
        return (
            sum(m.x[v, i, k] for i in m.N if i != k)
            - sum(m.x[v, k, j] for j in m.N if j != k)
            == 0
        )
    m.FlowClient = Constraint(m.V, m.I, rule=flow_client_rule)

    # 3. Asignación de vehículo a un solo depósito si se usa
    def assign_depot_rule(m, v):
        return sum(m.z[v, d] for d in m.D) == m.y[v]
    m.AssignDepot = Constraint(m.V, rule=assign_depot_rule)

    # 4. Salidas y entradas en depósito según asignación
    def depot_out_rule(m, v, d):
        return sum(m.x[v, d, j] for j in m.I) == m.z[v, d]
    m.DepotOut = Constraint(m.V, m.D, rule=depot_out_rule)

    def depot_in_rule(m, v, d):
        return sum(m.x[v, i, d] for i in m.I) == m.z[v, d]
    m.DepotIn = Constraint(m.V, m.D, rule=depot_in_rule)

    # 5. Si hay arcos, el vehículo debe estar activo
    M_veh = 2 * len(nodes)
    def use_vehicle_rule(m, v):
        return sum(m.x[v, i, j] for i in m.N for j in m.N if i != j) <= M_veh * m.y[v]
    m.UseVehicle = Constraint(m.V, rule=use_vehicle_rule)

    # 6. Relación entre carga inicial, capacidad de vehículo y asignación
    def init_cap_vehicle_rule(m, v):
        return sum(m.Init[v, d] for d in m.D) <= m.Q[v] * m.y[v]
    m.InitCapVehicle = Constraint(m.V, rule=init_cap_vehicle_rule)

    def init_link_assign_rule(m, v, d):
        return m.Init[v, d] <= m.Q[v] * m.z[v, d]
    m.InitLinkAssign = Constraint(m.V, m.D, rule=init_link_assign_rule)

    # 7. Capacidad de inventario en cada depósito
    def depot_inventory_rule(m, d):
        return sum(m.Init[v, d] for v in m.V) <= m.CapDepot[d]
    m.DepotInventory = Constraint(m.D, rule=depot_inventory_rule)

    # 8. Carga en depósitos igual a carga inicial
    def init_load_at_depot_rule(m, v, d):
        return m.u[v, d] == m.Init[v, d]
    m.InitLoadAtDepot = Constraint(m.V, m.D, rule=init_load_at_depot_rule)

    # 9. Cota superior de carga en nodos
    def load_upper_rule(m, v, n):
        return m.u[v, n] <= m.Q[v]
    m.LoadUpper = Constraint(m.V, m.N, rule=load_upper_rule)

    # 10. Transición de carga en arcos que llegan a clientes
    def load_transition_rule(m, v, i, j):
        if i == j or (j not in m.I):
            return Constraint.Skip
        return m.u[v, j] <= m.u[v, i] - m.q[j] + m.Q[v] * (1 - m.x[v, i, j])
    m.LoadTrans = Constraint(m.V, m.N, m.N, rule=load_transition_rule)

    # 11. Restricción de autonomía (rango máximo por vehículo)
    def range_limit_rule(m, v):
        return sum(m.c[i, j] * m.x[v, i, j] for i in m.N for j in m.N if i != j) <= m.R[v]
    m.RangeLimit = Constraint(m.V, rule=range_limit_rule)

    # Expresiones de distancia, tiempo y costo de combustible
    def expr_dv(m, v):
        return sum(m.c[i, j] * m.x[v, i, j] for i in m.N for j in m.N if i != j)
    m.dv = Expression(m.V, rule=expr_dv)

    def expr_tv(m, v):
        return m.dv[v] / m.speed_kmph
    m.tv = Expression(m.V, rule=expr_tv)

    def expr_fuel_cost(m, v):
        return m.dv[v] * m.fuel_cost_per_km[v]
    m.FuelCost_v = Expression(m.V, rule=expr_fuel_cost)

    # Función objetivo
    def obj_total_cost(m):
        return sum(
            m.C_fixed * m.y[v] +
            m.C_dist * m.dv[v] +
            m.C_time * m.tv[v] +
            m.FuelCost_v[v]
            for v in m.V
        )
    m.TotalCost = Objective(rule=obj_total_cost, sense=minimize)

    # Guardar info auxiliar en el modelo para postproceso
    m.coords = coords
    m.client_ids = client_ids
    m.depot_ids = depot_ids
    m.vehicle_ids = vehicle_ids

    return m


# -----------------------------
# Resolución del modelo
# -----------------------------

def solve_model(m: ConcreteModel, solver_name: str = "glpk"):
    """
    Resuelve el modelo con el solver especificado.
    """
    solver = SolverFactory(solver_name)
    if solver is None:
        raise RuntimeError(f"No se pudo crear el solver '{solver_name}'. Revise la instalación.")

    # --------- ÚNICO CAMBIO: límite de tiempo 20 minutos (1200 segundos) ---------
    if solver_name.lower() == "glpk":
        solver.options["tmlim"] = 1200  # 20 minutos
    # -----------------------------------------------------------------------------    

    results = solver.solve(m, tee=True)
    return results



# -----------------------------
# Postproceso - reconstrucción de rutas
# -----------------------------

def reconstruir_ruta(m: ConcreteModel, v: str, depot_id: str) -> List[str]:
    """
    Reconstruye la secuencia de nodos para un vehículo v,
    empezando y terminando en depot_id.
    """
    succ = {}
    for i in m.N:
        for j in m.N:
            if i == j:
                continue
            val = m.x[v, i, j].value
            if val is not None and val > 0.5:
                succ[i] = j

    ruta = [depot_id]
    actual = depot_id
    visitados = {depot_id}
    max_steps = len(list(m.N)) + 5

    for _ in range(max_steps):
        if actual not in succ:
            break
        nxt = succ[actual]
        ruta.append(nxt)
        if nxt == depot_id:
            break
        if nxt in visitados:
            break
        visitados.add(nxt)
        actual = nxt
    return ruta


def generar_verificacion_caso2(m: ConcreteModel, output_path: str = "verificacion_caso2.csv") -> pd.DataFrame:
    """
    Genera el archivo verificacion_caso2.csv con las columnas:
    VehicleId, DepotId, InitialLoad, RouteSequence, ClientsServed,
    DemandsSatisfied, TotalDistance, TotalTime, FuelCost
    """
    rows = []

    demanda = {i: float(m.q[i]) for i in m.I}

    for v in m.V:
        if m.y[v].value is None or m.y[v].value < 0.5:
            continue

        depots_activos = [d for d in m.D if m.z[v, d].value is not None and m.z[v, d].value > 0.5]
        if not depots_activos:
            continue
        depot_id = depots_activos[0]

        ruta = reconstruir_ruta(m, v, depot_id)
        clientes_ruta = [n for n in ruta if n in m.I]

        init_load = sum(float(m.Init[v, d].value) for d in m.D if m.Init[v, d].value is not None)

        demandas_lista = [demanda[i] for i in clientes_ruta]
        clients_served = len(clientes_ruta)

        total_dist = 0.0
        for i, j in zip(ruta[:-1], ruta[1:]):
            total_dist += float(m.c[i, j])

        speed = float(m.speed_kmph)
        total_time_hours = total_dist / speed if speed > 0 else 0.0
        total_time_minutes = 60.0 * total_time_hours

        fuel_cost_km = float(m.fuel_cost_per_km[v])
        fuel_cost = total_dist * fuel_cost_km

        rows.append({
            "VehicleId": str(v),
            "DepotId": str(depot_id),
            "InitialLoad": init_load,
            "RouteSequence": "-".join(ruta),
            "ClientsServed": clients_served,
            "DemandsSatisfied": "-".join(str(d) for d in demandas_lista),
            "TotalDistance": total_dist,
            "TotalTime": total_time_minutes,
            "FuelCost": fuel_cost,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Archivo '{output_path}' generado correctamente.")
    return df


# -----------------------------
# Mapa opcional con Folium
# -----------------------------

def generar_mapa_caso2(m: ConcreteModel, output_path: str = "mapa_caso2.html"):
    """
    Genera un mapa HTML con las rutas del Caso 2 si folium está instalado.
    """
    try:
        import folium
    except ImportError:
        print("folium no está instalado. Se omite la generación del mapa.")
        return

    coords = m.coords

    coord_rows = []
    for n, (lat, lon) in coords.items():
        tipo = "client" if n in m.I else "depot"
        coord_rows.append({"id": n, "lat": lat, "lon": lon, "type": tipo})
    coord_df = pd.DataFrame(coord_rows)

    center_lat = coord_df["lat"].mean()
    center_lon = coord_df["lon"].mean()

    mapa = folium.Map(location=[center_lat, center_lon], zoom_start=11, control_scale=True)

    for _, row in coord_df.iterrows():
        if row["type"] == "depot":
            folium.Marker(
                location=(row["lat"], row["lon"]),
                popup=row["id"],
                icon=folium.Icon(color="red", icon="home"),
            ).add_to(mapa)
        else:
            folium.CircleMarker(
                location=(row["lat"], row["lon"]),
                radius=4,
                popup=row["id"],
            ).add_to(mapa)

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c",
        "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    color_for = {}
    for idx, v in enumerate(sorted(list(m.V))):
        color_for[v] = palette[idx % len(palette)]

    for v in m.V:
        if m.y[v].value is None or m.y[v].value < 0.5:
            continue
        depots_activos = [d for d in m.D if m.z[v, d].value is not None and m.z[v, d].value > 0.5]
        if not depots_activos:
            continue
        depot_id = depots_activos[0]
        ruta = reconstruir_ruta(m, v, depot_id)
        coords_route = [(coords[n][0], coords[n][1]) for n in ruta]

        import folium  # por seguridad si se usa en otro contexto
        folium.PolyLine(
            coords_route,
            weight=4,
            opacity=0.8,
            tooltip=f"Vehículo {v}",
            color=color_for[v],
        ).add_to(mapa)

    mapa.save(output_path)
    print(f"Mapa guardado como '{output_path}'.")


# -----------------------------
# Función principal
# -----------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Proyecto A - Caso 2 - Urban Logistics (Bogotá)")
    parser.add_argument("--data-dir", type=str, default="data", help="Carpeta donde están los CSV de entrada")
    parser.add_argument("--solver", type=str, default="glpk", help="Nombre del solver de Pyomo (por ejemplo glpk, cbc)")
    parser.add_argument("--no-map", action="store_true", help="No generar el mapa HTML de rutas")
    args = parser.parse_args()

    print("Leyendo datos desde:", args.data_dir)
    data = load_data(args.data_dir)

    print("Construyendo modelo de Pyomo para Caso 2...")
    m = build_model(data)

    print("Resolviendo modelo con solver:", args.solver)
    results = solve_model(m, solver_name=args.solver)
    print("Estado del solver:", results.solver.status)
    print("Condición de terminación:", results.solver.termination_condition)
    try:
        print("Valor de la función objetivo:", value(m.TotalCost))
    except Exception:
        print("No se pudo evaluar la función objetivo. Revise la solución del solver.")

    print("Generando archivo de verificación verificacion_caso2.csv...")
    output_verif = os.path.join(args.data_dir, "verificacion_caso2.csv")
    df_verif = generar_verificacion_caso2(m, output_path=output_verif)
    print(df_verif.head())

    if not args.no_map:
        print("Generando mapa HTML de rutas (si folium está instalado)...")
        output_map = os.path.join(args.data_dir, "mapa_caso2.html")
        generar_mapa_caso2(m, output_path=output_map)

    print("Ejecución completada.")


if __name__ == "__main__":
    main()
