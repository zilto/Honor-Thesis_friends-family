# Sampling
Accel : (x, y, z) sampled at 4Hz (initial study)
AccelAccum : every 2 minutes, for 15 second (or 20 sec before the app was updated), (x, y, z) is sampled at 5Hz. For each second, the vector magnitude and variance are computed to assess if the second is of "moderate" or "high" activity (fix threshold) (this replaces Accel in full study)
App : per scan : one row per app installed
AppRunning : per scan : one row per app running
Battery : opportunistic sampling based on state change (e.g., battery level, temperature, or volt change)
BluetoothProximity : per scan : one row per device detected
CallLog : call labeled as "incoming", "outgoing", or "missed" with duration; old labels (initial study) are "incoming+" and "outgoing+" have no associated duration
Location : sampled every ~30 min
SMSLog : SMS by timestamp, can be multiple rows (SMS) for a timestamp (resolution in seconds)
WiFi : per scan: one row per network detected
Labels
DailyEMA : contains target label for each day

# Documentation on accelerometer
Accelerometer scans were sampled in a duty cycle of 15 s every 2 min. During the 15 s, raw 3-axis accelerometer measurements are sampled at 5 Hz rate and combined to compute the vector magnitude for each sample. The variance of the magnitude in each onesecond block is then computed. The score was calculated by giving one point for every second, thresholded to three states (1) ‘‘still’’ (2) ‘‘moderate activity’’ (3) ‘‘high activity’’. However, in this paper we are interested in general change in activity levels and therefore combine the two active levels.

Acceleration of the phone (Accel.csv.bz2, AccelAccum.csv.bz2), with each record including
  the participant ID associated with the data-collecting mobile phone (ParticipantID),  
  the time of the record (scantime), and
  the x, y, z readings of the accelerometer, sampled 4 times per second (pre Oct. 2010), or 
  the number of samples with bigger acceleration, indication physical exercise, sampled once per minute (post Oct. 2010).
