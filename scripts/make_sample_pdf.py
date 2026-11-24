"""Generate data/sample.pdf (dev-only; needs fpdf2, see requirements-dev.txt)."""
from pathlib import Path

from fpdf import FPDF

PAGES: list[tuple[str, str]] = [
    (
        "Orion-7 Pump Controller: Overview",
        "The Orion-7 is a programmable pump controller used in municipal water "
        "treatment plants. It drives up to four variable-frequency pumps and "
        "reports status over Modbus RTU at 19200 baud. The main board carries "
        "part number PN-4471-B, and the removable I/O daughter card is PN-9902-C.\n\n"
        "Typical installations run the controller in AUTO mode, where flow "
        "setpoints are received from the plant SCADA system. In LOCAL mode, an "
        "operator uses the front panel keypad to set the pump speed directly. The "
        "controller stores the last 512 events in non-volatile memory, so history "
        "survives a power cycle.\n\n"
        "Rated supply voltage is 24 V DC with a tolerance of plus or minus 10 "
        "percent. Maximum current draw is 1.8 A with all four relays energised. "
        "The enclosure is rated IP65 and operates between -20 C and 60 C.",
    ),
    (
        "Installation and Wiring",
        "Mount the controller on a DIN rail inside a ventilated cabinet. Leave at "
        "least 40 mm of clearance above and below the unit. Route sensor cables "
        "away from motor leads to avoid induced noise.\n\n"
        "Terminal block T1 carries the 24 V DC supply: pin 1 is positive, pin 2 "
        "is negative, and pin 3 is the protective earth. Terminal block T2 hosts "
        "the RS-485 bus: A on pin 1, B on pin 2, and shield on pin 3. Terminate "
        "the bus with a 120 ohm resistor at both ends of the line.\n\n"
        "The analog pressure transmitter, part number PT-220-4A, connects to "
        "input AI-1 using a 4 to 20 mA loop. Use shielded twisted pair, and "
        "ground the shield at the controller end only. A flow meter of type FM-88 "
        "may be connected to the pulse input DI-3, which accepts up to 1 kHz.",
    ),
    (
        "Fault Codes and Diagnostics",
        "The front panel display shows a fault code whenever the controller "
        "detects an abnormal condition. The most common codes are listed here.\n\n"
        "Error E-4012: Suction pressure below minimum. The pressure transmitter "
        "on AI-1 reports less than 0.5 bar for more than 10 seconds. Check for a "
        "closed inlet valve or an air-locked pump. The pump is stopped to prevent "
        "dry running.\n\n"
        "Error E-4027: Motor overcurrent. The drive reported current above 115 "
        "percent of rated value for 3 seconds. Inspect the motor windings and the "
        "impeller for blockage.\n\n"
        "Error E-5103: Modbus timeout. No valid frame was received from the "
        "SCADA master within 30 seconds. Verify the RS-485 wiring, the baud rate, "
        "and the slave address, which defaults to 17.\n\n"
        "Error E-6200: Event log checksum mismatch. The non-volatile memory is "
        "corrupted. Contact support with the board part number PN-4471-B and the "
        "firmware version. Clearing the log with the CLR EVT menu item usually "
        "resolves the issue but erases history.",
    ),
    (
        "Firmware Updates",
        "Firmware is distributed as signed image files named "
        "orion7-fw-MAJOR.MINOR.PATCH.bin. The current stable release is 3.2.1, "
        "released after the 3.1.x series fixed a rounding bug in the flow "
        "totaliser.\n\n"
        "To update, copy the image to a FAT32 USB stick and insert it into the "
        "service port behind the front cover. Enter the SERVICE menu with the "
        "code 7-3-1-9, then choose FW UPDATE. The controller verifies the Ed25519 "
        "signature before flashing. The process takes about four minutes, and the "
        "pumps hold their last commanded speed during the update.\n\n"
        "Never remove power while the progress bar is visible. If an update is "
        "interrupted, the bootloader keeps the previous image and the controller "
        "starts in SAFE state, showing code W-0001. Repeat the update to recover. "
        "Downgrades below version 3.0.0 are not supported because of a changed "
        "event log format.",
    ),
    (
        "Calibration Procedure",
        "Calibrate the pressure input once a year, or after replacing the "
        "transmitter. The procedure needs a reference pressure source accurate to "
        "0.1 percent and a loop calibrator.\n\n"
        "Step 1: Put the controller in LOCAL mode and stop all pumps. Step 2: "
        "Open CALIB then AI-1. Step 3: Apply 0 bar and press ENTER to store the "
        "zero point; the display should read 4.00 mA. Step 4: Apply the "
        "full-scale pressure of 10 bar and press ENTER to store the span; the "
        "display should read 20.00 mA. Step 5: Verify linearity at 2.5, 5.0, and "
        "7.5 bar. Deviation must stay below 0.25 percent of span.\n\n"
        "If the deviation exceeds the limit, replace the transmitter PT-220-4A. "
        "Record the results in the maintenance log and attach the calibration "
        "certificate of the reference source.",
    ),
    (
        "Maintenance Schedule",
        "Perform the following tasks to keep the controller reliable.\n\n"
        "Monthly: inspect the cabinet for moisture and dust. Confirm that the "
        "status LED blinks green once per second. Download and archive the event "
        "log through the USB service port.\n\n"
        "Quarterly: tighten all terminal screws to 0.5 Nm. Test the emergency "
        "stop input by opening the circuit on DI-1 and confirming that all pumps "
        "stop within 200 ms. Check the battery-backed real-time clock; drift "
        "above 2 minutes per month means the CR2032 cell should be replaced.\n\n"
        "Yearly: run the calibration procedure, replace the cabinet air filter "
        "(part number AF-310), and review firmware release notes for security "
        "fixes. Keep at least one spare I/O card, PN-9902-C, on site because "
        "lead time is typically six weeks.",
    ),
    (
        "Warranty and Support",
        "The Orion-7 carries a 36 month limited warranty from the date of "
        "shipment. The warranty does not cover damage caused by incorrect wiring, "
        "surge events without protective devices, or unauthorised firmware.\n\n"
        "To open a support case, collect the serial number printed on the side "
        "label, the firmware version, and an export of the event log. Cases are "
        "classified by severity: S1 means a plant outage, answered within 2 "
        "hours; S2 means degraded operation, answered within one business day; "
        "S3 covers general questions, answered within three business days.\n\n"
        "Returned units need an RMA number, for example RMA-2025-0042, issued by "
        "support. Pack the unit in anti-static material and include a short "
        "description of the fault. Repairs typically take ten working days.",
    ),
    (
        "Troubleshooting in Plain Language: Unexpected Stops",
        "Operators often report that the water-moving machinery halts on its own "
        "when the supply reservoir runs close to dry. This is a protective "
        "behaviour, not a defect. The controller watches the pressure on the "
        "incoming side of the line, and when that reading sinks beneath a safe "
        "floor for a few seconds, it concludes that the machinery is starving "
        "for liquid. Running without liquid overheats the seals within minutes "
        "and can ruin the rotating assembly, so the safest response is to "
        "shut the drive down and wait.\n\n"
        "To resolve the situation, first let the reservoir refill to a healthy "
        "level. Then confirm that the valve on the incoming side is fully open, "
        "because a partly closed valve mimics a low supply. Look for trapped "
        "air in the suction pipe: a gurgling sound or a pressure needle that "
        "flutters is a typical sign. Bleed the air through the vent plug until "
        "a steady stream emerges. Finally, check that the level sensor in the "
        "reservoir is clean, since a coating of sediment can make it report "
        "less liquid than is really present.\n\n"
        "If the stoppages continue even though the reservoir is full, the "
        "pressure sensor itself may be drifting. Compare its reading against a "
        "trusted gauge fitted next to it. A persistent difference means the "
        "sensor needs recalibration or replacement. Also review how long the "
        "controller waits before reacting; a delay that is too short makes the "
        "system jumpy during brief dips caused by turbulence.",
    ),
    (
        "Troubleshooting in Plain Language: Noise and Vibration",
        "A healthy installation hums quietly and steadily. When the sound turns "
        "into a rattle, a growl or a high whine, something is wrong in the "
        "mechanical path or in the way liquid reaches the impeller.\n\n"
        "A crackling noise, as if gravel were being thrown around inside the "
        "housing, usually indicates that tiny vapour bubbles form on the "
        "intake side and then collapse violently. This effect, known as "
        "cavitation, erodes metal surfaces over time. It is caused by an "
        "intake that is too restricted, a liquid that is too warm, or a "
        "machine asked to move more volume than the pipework can deliver. "
        "Reduce the requested speed, open restrictions, or lower the "
        "temperature of the liquid.\n\n"
        "A rhythmic thumping that rises and falls with rotation points to a "
        "worn bearing or a shaft that is slightly bent. Stop the unit, turn "
        "the shaft by hand, and feel for roughness or a tight spot. "
        "Loose mounting bolts also amplify vibration, so check the base plate "
        "and the coupling guard. A shaky pipe connected rigidly to the unit "
        "can carry vibration into the whole cabinet, so use flexible joints "
        "where the pipe meets the casing.\n\n"
        "Whenever the noise is new, write down the speed, the temperature and "
        "the time of day. Patterns in these notes help the support team find "
        "the cause much faster than a general description.",
    ),
    (
        "Concepts: Control Modes Explained",
        "Think of the controller as a careful assistant standing beside the "
        "machinery. In automatic operation, the assistant follows instructions "
        "that arrive from the central supervisory computer: how much liquid to "
        "move, when to ramp up, and when to rest. The operator does not need to "
        "touch anything, and the plant can coordinate many stations at once.\n\n"
        "In manual operation, the person at the cabinet takes over. This is "
        "useful during commissioning, cleaning or an emergency, because the "
        "assistant stops listening to the remote computer and obeys only the "
        "buttons on the front. The danger is forgetting to hand control back, "
        "which leaves the station deaf to the plant schedule. A good habit is "
        "to attach a visible tag to the door whenever manual operation is "
        "chosen and to remove it only after returning to automatic.\n\n"
        "A third idea is the soft ramp. Rather than jumping straight to the "
        "requested speed, the drive climbs gradually. Gentle ramps protect "
        "pipes from pressure spikes, sometimes called water hammer, and reduce "
        "wear on couplings. Ramp times are a trade-off: very slow ramps waste "
        "time, while very fast ramps stress the hydraulics.\n\n"
        "Finally, the idea of redundancy deserves mention. When several units "
        "share a duty, the controller can rotate which one leads, so that "
        "wear is spread evenly. If one unit fails, the others automatically "
        "absorb its share until a technician arrives.",
    ),
    (
        "Concepts: Sensors and Measurements",
        "Every decision made by the controller rests on measurements. A "
        "pressure transducer converts force on a thin membrane into a small "
        "electrical current; a higher force produces a larger current. The "
        "controller reads this current and translates it back into a pressure "
        "figure. Because the lowest current represents zero pressure, a broken "
        "wire is easy to recognise: the current falls below the lowest valid "
        "value, which no real measurement can produce.\n\n"
        "Flow is estimated differently. A small wheel or an electronic probe "
        "creates a train of pulses, one for every small portion of liquid that "
        "passes. Counting pulses per second reveals how fast the liquid moves, "
        "and adding them up over time reveals the total volume delivered. "
        "Pulse counting is robust but loses accuracy at very low velocities, "
        "where the wheel barely turns.\n\n"
        "Temperature and level sensors complete the picture. Temperature "
        "tells whether motors are overheating; level tells whether the supply "
        "is sufficient. Sensors age, collect dirt and drift, so a comparison "
        "with an independent reference at regular intervals is the cornerstone "
        "of trustworthy automation.\n\n"
        "A useful rule of thumb is to distrust a single reading that "
        "contradicts everything else. If one value jumps while all related "
        "values stay calm, suspect the instrument or its cable before "
        "suspecting the process.",
    ),
    (
        "Operating Guidance: Seasonal and Environmental Factors",
        "Cold weather thickens liquids and makes seals stiffer. In freezing "
        "conditions, water trapped in pipes can expand and crack housings, so "
        "drain idle lines or keep a trickle moving. Cabinet heaters prevent "
        "condensation on circuit boards, which is a quiet but frequent cause "
        "of intermittent faults.\n\n"
        "Hot weather has the opposite effect. Electronics lose some of their "
        "margin as they warm up, and a cabinet in direct sunshine can become "
        "far hotter than the surrounding air. Shade the enclosure, keep vents "
        "unobstructed, and consider a small fan with a filter. A cabinet that "
        "feels uncomfortably hot to the hand is already too hot for long-term "
        "reliability.\n\n"
        "Humidity and dust bring their own troubles. Moist air encourages "
        "corrosion on terminals, producing resistance that grows slowly until "
        "a connection becomes unreliable. Fine dust forms an insulating blanket "
        "on components and traps heat. Wipe surfaces with a dry cloth during "
        "routine visits and never use a wet cloth inside a live cabinet.\n\n"
        "Lightning and nearby heavy machinery produce electrical disturbances. "
        "Good grounding, short cable runs and surge protection devices "
        "dramatically reduce the number of mysterious restarts that occur "
        "during storms.",
    ),
    (
        "Operating Guidance: Energy Efficiency",
        "Moving liquid consumes a large share of the electricity used by a "
        "treatment facility, so even small improvements pay off. The most "
        "effective trick is to match speed to demand. Running a drive at "
        "eighty percent speed uses roughly half the energy of full speed, "
        "because power rises with the cube of the rotational rate. Variable "
        "speed control therefore beats throttling a valve, which wastes energy "
        "as heat and noise.\n\n"
        "Scheduling matters too. Many utilities charge less for electricity "
        "at night, so filling storage during cheap hours and drawing it down "
        "during expensive ones reduces the bill without hurting service. "
        "The controller can follow such a timetable automatically if the "
        "supervisory system supplies it.\n\n"
        "Keep an eye on the operating point. Equipment is most efficient near "
        "the middle of its designed range, and efficiency drops sharply when "
        "it is forced to work far below or above that zone. If a unit "
        "constantly runs in a poor zone, a different size or an impeller "
        "trim may be a better investment than any amount of tuning.\n\n"
        "Leaks are silent energy thieves. A small hole in a pipe makes the "
        "machinery work longer to deliver the same useful volume. Regular "
        "inspections of joints and gaskets recover that wasted effort.",
    ),
    (
        "Training Notes: Safe Working Practices",
        "Before opening any cabinet, isolate the supply and prove that it is "
        "dead with a tester that you have checked on a known live source. "
        "Lock the isolator and keep the key in your pocket so nobody can "
        "re-energise the circuit while hands are inside.\n\n"
        "Rotating equipment can start unexpectedly if a remote command "
        "arrives. Always switch to manual operation, disable remote start, "
        "and place a warning tag before working on the mechanical side. "
        "Wear safety footwear, eye protection and close-fitting clothing; "
        "loose sleeves and jewellery can be caught by moving parts.\n\n"
        "Wet locations are especially hazardous. Use equipment rated for "
        "damp environments, keep leads off the floor and never touch metal "
        "parts with wet hands. When lifting heavy components, use a hoist "
        "or ask a colleague; a strained back ends more careers than any "
        "machine.\n\n"
        "After finishing, clear tools from the area, restore guards, remove "
        "tags and locks in the reverse order of fitting, and tell the control "
        "room that the station is ready to return to service. A short "
        "written handover note prevents misunderstandings between shifts.",
    ),
    (
        "Training Notes: Record Keeping and Handover",
        "Good records turn a confusing failure into a quick repair. Each "
        "station should have a logbook, paper or electronic, that captures "
        "what happened, when, who noticed and what was done. Entries should "
        "be factual: record readings and observations rather than guesses.\n\n"
        "When a fault occurs, capture a snapshot of the situation before "
        "touching anything. Photograph the display, note the position of "
        "switches and valves, and save the recent history from the "
        "controller. Evidence disappears quickly once someone starts "
        "adjusting settings, and the first few minutes are often the most "
        "informative.\n\n"
        "At shift changes, spend five minutes talking through open issues. "
        "Mention anything that behaves strangely even if it seems minor; "
        "intermittent oddities are often early warnings of larger trouble. "
        "Keep a list of spare items on site, with quantities and storage "
        "locations, so that the next person does not waste time searching.\n\n"
        "Review the logbook monthly with the team. Repeated entries about the "
        "same symptom reveal a design weakness or a training gap that "
        "deserves a permanent fix instead of yet another temporary remedy.",
    ),
    (
        "Concepts: Why Alarms Are Graded",
        "Not every warning deserves the same urgency. The controller sorts "
        "its messages into three grades so that people can react in proportion. "
        "Information messages simply record that something happened, such as "
        "a completed update or a changed setting. Nobody needs to rush to the "
        "cabinet for these; they are read later during a routine visit.\n\n"
        "Warnings sit in the middle. They signal that a value is drifting "
        "toward a limit, for example a motor running warmer than usual or a "
        "communication link that drops a few messages. The plant is still "
        "working, but a technician should look at the cause within days, "
        "before a minor irritation grows into a stoppage.\n\n"
        "Critical alarms demand immediate attention because equipment or "
        "people may be at risk, or because the process has already halted. "
        "These alarms also trigger the protective actions built into the "
        "controller, such as ramping the drive down or opening a bypass.\n\n"
        "If everything is critical, nothing is. Reviewing alarm statistics "
        "every quarter helps to remove noisy messages that operators have "
        "learned to ignore, so that the truly important ones stand out on the "
        "screen and in the minds of the people on duty.",
    ),
    (
        "Concepts: Communication Between Stations",
        "Stations talk to the central computer over a shared two-wire line. "
        "Imagine a meeting in which a chairperson asks each participant a "
        "question in turn; only the participant addressed may answer. That "
        "is how the line stays orderly even though dozens of devices are "
        "connected to the same cable.\n\n"
        "Reliability depends on the physical layer. Cables should be run in "
        "one continuous chain rather than in a star, because branches cause "
        "reflections that garble the signal, much like echoes in a tunnel. "
        "Both ends of the chain need a matching resistor to absorb the signal "
        "and prevent such echoes.\n\n"
        "Every participant needs a unique address and the same speed setting "
        "as the chairperson. A single device configured with a different "
        "speed can disturb the whole line by producing gibberish that others "
        "mistake for part of a conversation. When the line misbehaves, "
        "disconnect devices one at a time until the noise vanishes.\n\n"
        "Silence is also information. If the chairperson stops asking "
        "questions for too long, each station assumes the link was lost and "
        "falls back to a safe behaviour that was chosen during setup.",
    ),
    (
        "Operating Guidance: Start-Up Checklist",
        "Before the first start of a new installation, walk through the "
        "following sequence calmly and tick off each step. Confirm that "
        "the wiring matches the drawings and that every terminal screw is "
        "tight. Verify that the supply voltage lies inside the allowed range "
        "and that polarity is correct, since a reversed connection can "
        "damage the input stage.\n\n"
        "Next, fill the pipework slowly and vent trapped air at every high "
        "point. Turn the shaft by hand to ensure it moves freely. Check the "
        "direction of rotation with a brief jog: the arrow cast on the "
        "housing shows the correct direction, and running backwards "
        "reduces delivery and can unscrew the impeller.\n\n"
        "Then enter the basic settings: the speed limits, the ramp times, "
        "the protection thresholds and the communication address. Save the "
        "settings and note them in the logbook. Run the station in manual "
        "operation for fifteen minutes while watching currents, temperatures "
        "and noise.\n\n"
        "Only after everything looks stable should the station be handed to "
        "the remote computer. Observe the first automatic cycle personally, "
        "and keep a hand near the emergency stop until it has completed "
        "without surprises.",
    ),
    (
        "Operating Guidance: Spare Parts Strategy",
        "A sensible spare parts policy balances the cost of stock against "
        "the cost of waiting. Items that fail often and cost little, such as "
        "fuses, seals and filters, should be kept in generous quantities "
        "at every site. Items that fail rarely but are expensive, such as "
        "complete boards, are better held at a central warehouse that serves "
        "several stations.\n\n"
        "Lead time matters as much as price. A part that can be delivered "
        "overnight needs far less buffer than one built to order over "
        "several weeks. For the long-lead items, keep at least one unit on "
        "the shelf at all times, and reorder the moment it is used.\n\n"
        "Store electronic parts in anti-static bags, in a dry cabinet, and "
        "away from strong magnets and direct sunshine. Label each box with "
        "the description, the date received and the storage location. "
        "Rotate stock so that older items are used first; some components, "
        "especially batteries and capacitors, deteriorate on the shelf.\n\n"
        "Once a year, test a sample of the stored spares on a bench to "
        "confirm that they still work. Discovering a dead spare during an "
        "emergency is a particularly unpleasant experience.",
    ),
    (
        "Training Notes: Reading Trends and Logs",
        "Numbers recorded over time tell stories that a single reading "
        "cannot. A motor current that creeps upward by a small amount each "
        "week suggests growing friction, perhaps from a wearing bearing or "
        "a fouled impeller. A pressure that oscillates in a regular rhythm "
        "suggests a control loop that is tuned too aggressively and keeps "
        "overcorrecting.\n\n"
        "When examining a trend, first look at the scale. A graph that "
        "appears dramatic may span only a tiny range, while a flat line may "
        "hide a significant change if the scale is too coarse. Next, "
        "compare related signals on the same time axis. If the current "
        "rises just before the pressure falls, the two events probably share "
        "a cause.\n\n"
        "Look for repetition. Events that occur at the same time every "
        "day often connect to schedules, such as a neighbouring plant "
        "starting its machines or a storage basin being drawn down for cleaning. "
        "Events that occur at random intervals lean toward loose "
        "connections or interference.\n\n"
        "Finally, annotate the graphs. A short note stating what was done "
        "and when turns a puzzling kink in the line into an explained, "
        "harmless event for whoever reads the history next year.",
    ),
    (
        "Training Notes: Common Misconceptions",
        "A frequent belief is that a higher setting for speed always yields "
        "a better result. In reality, pushing equipment beyond its efficient "
        "range increases wear, noise and energy cost while delivering only a "
        "small gain in volume. The best setting is the lowest one that meets "
        "the demand.\n\n"
        "Another misconception is that a fault which disappears after a "
        "restart has been fixed. Restarting clears the symptom but not the "
        "cause. If the same message returns, treat it as a real problem and "
        "investigate rather than resetting repeatedly.\n\n"
        "Some people assume that sensors never need attention because they "
        "have no moving parts. Yet membranes fatigue, cables corrode and "
        "reference points drift. Regular comparison against a trusted "
        "instrument is the only reliable way to know that the number on the "
        "screen is honest.\n\n"
        "Finally, many newcomers think that documentation is a chore to be "
        "completed after the real work. Experienced technicians know that "
        "clear notes save more time than any tool, because they allow the "
        "next person to continue exactly where the previous one stopped.",
    ),
    (
        "Concepts: Protection Against Overload",
        "Every motor has a limit to how much twisting force it can deliver "
        "without overheating. The controller guards that limit by watching "
        "the electrical current, which rises when the load grows heavier. "
        "A brief surge during start-up is normal and tolerated; a surge that "
        "lasts too long is treated as a sign of trouble.\n\n"
        "Typical causes of heavy load include debris wedged in the moving "
        "parts, a bearing that has started to seize, liquid that is thicker "
        "than expected, or a pipe downstream that has been closed by "
        "mistake. In each case the drive works harder to achieve the same "
        "result, and the windings warm up quickly.\n\n"
        "When the protective limit is exceeded, the controller first reduces "
        "speed to relieve the strain. If the current stays high, it stops "
        "the drive and records the event. Waiting for the motor to cool "
        "before another attempt prevents permanent damage to the insulation "
        "of the windings, which is expensive to repair.\n\n"
        "Technicians should resist the temptation to raise the protective "
        "limit simply to stop nuisance stoppages. A higher limit hides the "
        "symptom while the underlying problem continues to wear the "
        "equipment, and the eventual failure will be far more costly.",
    ),
]


def main() -> None:
    """Write the sample PDF into the data directory."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for title, body in PAGES:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.multi_cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, body)
    out = Path(__file__).resolve().parent.parent / "data" / "sample.pdf"
    out.parent.mkdir(exist_ok=True)
    pdf.output(str(out))
    print(f"wrote {out} ({len(PAGES)} pages)")


if __name__ == "__main__":
    main()
