[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/IuXd4k6Y)
# Project 3 — Flight Route & Fare Comparator

*Working title: “FlyWise: Smart Route & Fare Planner”*

Your team is building the core engine of a tiny “Google Flights–style” planner. Given a small network of airports and scheduled flights, your program will:

* Read a list of flights (airports, times, and prices per cabin).
* Build an internal flight graph.
* Answer a **comparison query** for a route:

  * Find the **earliest-arrival itinerary**.
  * Find the **cheapest Economy itinerary**.
  * Find the **cheapest Business itinerary**.
  * Find the **cheapest First itinerary**.
* Print a clean text **comparison table** of those itineraries.

Every itinerary must obey **time** and **layover** rules. You must use both **graphs** and **hash tables (dicts)** in your design.

You are **not** building a real API or website. This is a **Python CLI program** with a clean interface and solid data structures.

---

## 0. Learning goals

By the end of this project, you should be able to:

1. Model a real-world system (airline routes) as a **graph**.
2. Store that graph efficiently using **adjacency lists** in Python.
3. Use **hash tables (dicts)** for:

   * Fast lookup of airports, flights, and best-known values.
   * Caching / memoization where appropriate.
4. Implement and reason about a **shortest-path–style search** on a graph where the “cost” can be:

   * Time (earliest arrival).
   * Money (cheapest price in a given cabin).
5. Enforce real-world constraints during graph search:

   * Minimum layover times.
   * Chronological order (no time-travel flights).
6. Explain **time and space complexity** for your core operations.
7. Design a small but readable **CLI** that exposes the core functionality clearly.

You will practice:

* Translating a fuzzy product requirement (“compare routes”) into concrete data structures.
* Gradually building up functionality: parsing → modeling → searching → formatting output.

---

## 1. Problem story

An imaginary startup, **FlyWise**, wants a minimal but smart route engine for a demo to investors. They don’t care about real-world data, fancy UIs, or booking. They just want to show:

> “Given some routes and prices, our planner can find different ‘best’ itineraries and explain the tradeoffs in a single comparison chart.”

They’ve asked you to build the backend engine as a Python program.

* The engine reads a **flight schedule file**.
* Then it answers **comparison queries** like:
  `ICN → SFO, leaving at or after 08:00`.
* For each query, the program prints a simple text table summarizing the best itineraries under different criteria.

The investors don’t care how you do it internally, but **we do**: you must use graphs and hash tables, and you must respect time + layover constraints.

---

## 2. Data model

### 2.1 Airports & flights

We are modeling a **single day** of flights.

* Each **airport** has a short code (like `ICN`, `SFO`, `LAX`, `NRT`).
* Each **flight** is a directed edge: it goes from **one airport** to **another**.

Each flight record will have at least:

* `origin` — airport code (string)
* `destination` — airport code (string)
* `flight_number` — unique-ish id (string)
* `departure` — time of day (minutes since midnight, integer)
  (Example: `8:30` → `510`)
* `arrival` — time of day (minutes since midnight, integer)
* `economy_price` — integer (e.g., 520)
* `business_price` — integer (e.g., 1280)
* `first_price` — integer (e.g., 2440)

Note: You can assume **all flights start and finish on the same day**.

### 2.2 Input file formats

You will be given **two example schedule files** describing the same set of flights:

1. A **plain text** file with space-separated fields (the format described below).
2. A **CSV** file with the same columns but comma-separated.

You may choose **either format** for your implementation, or support **both** if you want the extra practice. The autograder will clearly state which format it uses.

#### 2.2.1 Plain text format (space-separated)

**Required format (one flight per line):**

```text
ORIGIN DEST FLIGHT_NUMBER DEPART ARRIVE ECONOMY BUSINESS FIRST
```

* Fields are separated by one or more spaces.
* Times are given as `HH:MM` in 24h format.
* Prices are integers.

**Example:**

```text
ICN NRT FW101 08:30 11:00 400 900 1500
NRT SFO FW202 12:30 06:30 500 1100 2000
ICN SFO FW303 13:00 20:30 900 1800 2600
SFO LAX FW404 21:30 23:00 120 300 700
```

You should:

* Ignore empty lines and lines starting with `#` (comments).
* Parse `HH:MM` into minutes since midnight (`int`).
* Convert each line into a `Flight` object or similar structure.

#### 2.2.2 CSV format (comma-separated)

The CSV file contains the **same columns**, but separated by commas and with a header row. For example:

```csv
origin,dest,flight_number,depart,arrive,economy,business,first
ICN,NRT,FW101,08:30,11:00,400,900,1500
NRT,SFO,FW202,12:30,06:30,500,1100,2000
ICN,SFO,FW303,13:00,20:30,900,1800,2600
SFO,LAX,FW404,21:30,23:00,120,300,700
```

You may:

* Use Python's built-in `csv` module, or
* Manually split on commas if you prefer.

The parsing rules are otherwise the same as for the plain text file:

* Times are `HH:MM` 24h.
* Prices are integers.
* All flights occur within a single day.

#### 2.2.3 Hint: unifying both formats

If you decide to support both input formats, a clean approach is:

* Write **two small loader functions**, for example:

  * `load_flights_txt(path: str) -> list[Flight]`
  * `load_flights_csv(path: str) -> list[Flight]`
* Have each one return the same internal `Flight` objects.
* Then write a tiny wrapper, e.g. `load_flights(path)` that:

  * Checks the file extension (`.txt` vs `.csv`), or
  * Peeks at the first line (does it look like a CSV header?).
  * Calls the appropriate loader.

The rest of your program (graph building, search) should **not care** which file type was used; it just receives a `list[Flight]`.

---

## 3. Required data structures

You must use both **graphs** and **hash tables (Python dict)**.

You have design freedom, but your implementation **must** satisfy these constraints:

### 3.1 Graph representation

* Represent airports as **nodes** in a directed graph.

* Represent flights as **directed edges**.

* Use an **adjacency list**–style structure, for example:

  ```python
  # Example idea (you may design your own):
  flights_from: dict[str, list[Flight]]
  # flights_from["ICN"] is a list of Flight objects departing ICN.
  ```

* This graph should store **all flights** in the input file.

You may define a `Flight` class/dataclass/NamedTuple if you like. That’s encouraged for clarity.

### 3.2 Hash tables (dicts)

Use dictionaries for at least **two** of the following (most solutions will naturally use more):

* `flights_from: dict[str, list[Flight]]` — adjacency list.
* `airport_codes: dict[str, AirportInfo]` — optional metadata about each airport.
* `best_time: dict[str, int]` — earliest known arrival time at each airport during a search.
* `best_cost: dict[str, int]` — cheapest known cost to each airport during a search.
* `previous: dict[str, Flight]` — for reconstructing itineraries.
* Any memoization cache you decide to add.

In your **README**, you will briefly describe **which dicts you used and why**.

---

## 4. Itineraries and constraints

An **itinerary** is a sequence of one or more flights:

```text
ICN --(FW101)--> NRT --(FW202)--> SFO
```

It must satisfy:

1. **Chronological order**
   For each consecutive pair of flights:

   * The arrival airport of flight `i` = the departure airport of flight `i+1`.
   * The next departure time is **on or after** the arrival time of the previous flight, plus a required layover.

2. **Minimum layover time**
   Let `MIN_LAYOVER_MINUTES` be a constant (for example, `60`). Then for each connection:

   ```text
   next_departure_time >= previous_arrival_time + MIN_LAYOVER_MINUTES
   ```

3. **Same-day assumption**
   All times are minutes from midnight of the same day.

You should define a Python representation of an itinerary, for example:

```python
class Itinerary:
    flights: list[Flight]
```

You will need at least these operations on an itinerary:

* Compute overall **departure time** (from first flight).
* Compute overall **arrival time** (from last flight).
* Compute **total duration** (arrival - departure).
* Compute **total price for a given cabin** (sum of economy/business/first prices across flights).
* Compute number of **stops** (flights - 1).

You may add helper methods or free functions to keep this tidy.

---

## 5. Core queries & CLI

Your program should be run from the command line. At minimum, it must support:

```bash
python flight_planner.py compare FLIGHT_FILE ORIGIN DEST DEPARTURE_TIME
```

Where:

* `FLIGHT_FILE` — path to the input file (as above).
* `ORIGIN` — starting airport code, e.g. `ICN`.
* `DEST` — destination airport code, e.g. `SFO`.
* `DEPARTURE_TIME` — earliest allowed departure time, in `HH:MM` 24h format.

Example:

```bash
python flight_planner.py compare flights_small.txt ICN SFO 08:00
```

### 5.1 Comparison table output

For each `compare` command, your program should:

1. Read the flights from `FLIGHT_FILE` and build your graph.
2. Run **four** separate searches (see next section):

   * Earliest-arrival itinerary (any cabin prices allowed, but comparisons are based on time).
   * Cheapest Economy itinerary.
   * Cheapest Business itinerary.
   * Cheapest First itinerary.
3. Print a formatted text table with **one row per mode**, including at least:

   * Mode name.
   * Cabin class used.
   * Overall departure time.
   * Overall arrival time.
   * Total duration.
   * Number of stops (0 = direct).
   * Total price.

You may choose your own layout and exact headings, but it should be **readable and aligned**.

**Example (conceptual only, not prescribed):**

```text
Comparison for ICN → SFO (earliest departure 08:00, layover ≥ 60 min)

Mode                    Cabin     Dep    Arr    Duration  Stops  Total Price
----------------------  --------  -----  -----  --------  -----  -----------
Earliest arrival        Economy   09:15  16:40  14h25m    1      780
Cheapest (Economy)      Economy   10:30  19:10  15h40m    2      520
Cheapest (Business)     Business  09:15  16:40  14h25m    1      1480
Cheapest (First)        First     12:00  22:05  18h05m    2      2480
```

If **no valid itinerary** exists for a mode (for example, no First-class flights connecting `ICN` to `SFO` after the given time), print something clear like:

```text
Cheapest (First)        First     N/A    N/A    N/A       N/A    N/A  (no valid itinerary)
```

You are not graded on ANSI colors or fancy formatting, only clarity.

---

## 6. Search behavior (high level)

You need to implement **searches on your flight graph** that obey the layover and timing rules.

You’ll implement at least two conceptual search modes:

### 6.1 Earliest-arrival search

Given:

* Start airport `S`.
* Destination airport `T`.
* Earliest allowed departure time `t0` (minutes since midnight).

Find: an itinerary from `S` to `T` that **arrives as early as possible**, subject to:

* Every flight departs from your current airport.
* Every connection respects `MIN_LAYOVER_MINUTES`.
* Each flight departs at or after your current time (plus layover for connections).

The **cost** you care about is **arrival time at the destination**.

You should write a function/method like:

```python
find_earliest_itinerary(graph, start, dest, earliest_departure) -> Itinerary | None
```

You are not required to use any particular algorithm by name in code, but you should be able to **describe** it in terms of **time/space complexity** in your README.

### 6.2 Cheapest itinerary for a cabin

Given:

* Start airport `S`.
* Destination airport `T`.
* Earliest allowed departure time `t0`.
* Cabin class: one of `"economy"`, `"business"`, `"first"`.

Find: an itinerary from `S` to `T` that has the **lowest total price** in that cabin, subject to the same timing and layover rules as above.

You should write something like:

```python
find_cheapest_itinerary(graph, start, dest, earliest_departure, cabin) -> Itinerary | None
```

Each flight contributes a price based on the chosen cabin, and you sum them.

Because you still enforce times and layovers, your search must **discard** impossible connections even if they would be cheap.

**Note:** There can be tradeoffs: the cheapest itinerary may take longer or involve more stops.

### 6.3 Complexity expectations

You must:

* Provide a **brief argument** in your README for the time and space complexity of:

  * Building the graph from the file.
  * Running one earliest-arrival search.
  * Running one cheapest-itinerary search.
* Use **Big-O notation** (e.g., `O(E log V)`) and explain what `V` and `E` mean.

We will not grade micro-optimizations, but your approach should be clearly better than a naive “brute force over all possible itineraries” search.

---

## 7. Edge cases & error handling

Your program should handle the following situations gracefully:

1. **Unknown airport codes**
   If `ORIGIN` or `DEST` is not present in the flight file, print a helpful error message and exit.

2. **No valid itinerary**

   * For earliest arrival: if you cannot reach the destination given the starting time and layover rules, indicate that no route exists.
   * For a specific cabin: if no route exists that has valid flights in that cabin for all legs, mark that mode as unavailable.

3. **No flights after the requested departure time**
   If no flights from the origin depart at or after `DEPARTURE_TIME`, then obviously no itinerary can exist; handle this without crashing.

4. **Invalid time format**
   If `DEPARTURE_TIME` is not a valid `HH:MM`, show a clear error message.

5. **Empty or malformed lines in the input file**

   * Ignore blank lines.
   * Ignore comment lines starting with `#`.
   * If a line is badly formatted (wrong number of fields), you may either:

     * Skip it with a warning, or
     * Exit with an error message.

You do **not** need to support:

* Multiple days / time zones.
* Airports with zero outbound flights (beyond normal behavior).

---

## 8. File organization & API expectations

### 8.1 Code files

At minimum, include:

* `flight_planner.py` — entry point that parses CLI arguments and prints results.
* One or more modules for:

  * Data models (Flight, Itinerary).
  * Graph building.
  * Search algorithms.

You may organize these however you like, but keep it **readable and modular**.

### 8.2 Tests

You will receive (or write) **pytest tests** that call your functions directly. To keep grading easier:

* Expose your core functions with clear signatures, for example:

  * `load_flights(path: str) -> list[Flight]`
  * `build_graph(flights: list[Flight]) -> GraphType`
  * `find_earliest_itinerary(...) -> Itinerary | None`
  * `find_cheapest_itinerary(...) -> Itinerary | None`
  * `format_comparison_table(...) -> str`

We will not test your CLI parsing deeply; we care mostly about the core logic.

---

## 9. Complexity & README requirements

Your README must include:

1. **High-level design**

   * How you represent the graph (what are the nodes/edges?).
   * Which dicts (hash tables) you use and what they map.

2. **Complexity analysis**
   For each of the following, give Big-O time and space, and a one- or two-sentence justification:

   * Building the graph from `N` flight records.
   * One earliest-arrival search on a graph with `V` airports and `E` flights.
   * One cheapest-itinerary search.

3. **Edge-case checklist**
   Bullet list of the edge cases from section 7, plus any others you thought about.

4. **How to run**

   * Sample commands.
   * Expected input file format (you can copy/adapt from this spec).

---

## 10. Rubric sketch (for transparency)

This is **not** the official grading rubric, but it hints at how your work will be evaluated.

1. **Correctness (core behavior)**

   * Reads the flight file correctly.
   * Builds a reasonable graph.
   * Produces valid earliest-arrival itineraries that respect layovers.
   * Produces valid cheapest itineraries per cabin.
   * Handles “no route” / “no cabin” cases cleanly.

2. **Data structures & algorithms**

   * Uses an adjacency-list graph.
   * Uses dictionaries clearly and appropriately.
   * Search is better than brute-force enumeration.

3. **Complexity reasoning**

   * README explains the complexity in Big-O.
   * Claims match the actual approach.

4. **Code quality**

   * Clear function decomposition.
   * Good naming and comments/docstrings.
   * No obvious dead code or copy-paste.

5. **CLI & output**

   * `compare` command works as specified.
   * Comparison table is readable.
   * Error messages make sense.

---

## 11. Suggested development steps (not graded)

You don’t have to follow this exactly, but here’s a low-stress path through the project:

1. **Parsing only**

   * Implement `parse_time`, `parse_flight_line`, and one or more `load_flights` helpers.
   * Decide whether you will support the plain text format, CSV, or both.
     *Hint:* convert both into the same internal `Flight` objects so the rest of your code doesn't need to know which format was used.
   * Print out a couple of flights to manually check.

2. **Graph building**

   * Build `flights_from` adjacency dict.
   * Add a helper to list all outgoing flights from a given airport.

3. **Itinerary type + basic printing**

   * Define an `Itinerary` class or simple structure.
   * Create one by hand (a couple of flights) and write a function to print it nicely.

4. **Earliest-arrival search**

   * Implement `find_earliest_itinerary`.
   * Test it on a tiny hard-coded graph before using file input.

5. **Cheapest-itinerary search**

   * Implement `find_cheapest_itinerary` for one cabin.
   * Generalize to any of `"economy"`, `"business"`, `"first"`.

6. **Comparison table**

   * Write `format_comparison_table` that accepts up to 4 itineraries and returns a string.
   * Plug it into the CLI.

7. **Edge cases + cleanup**

   * Add checks for unknown airports, no flights, bad times.
   * Polish README and complexity notes.

If you break your work into these steps and test each piece as you go, you’ll avoid the classic “it almost works but I don’t know why it’s broken” crunch.

Good luck, and may your layovers always be just long enough.
