import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle

# --- Global Variables for Matplotlib Text Objects ---
angle_range_text_obj = None
hover_tooltip_text = None
status_text_obj = None # Added for consistency, will be assigned later

# --- Conversion Factors ---
MPS_TO_MPH = 2.23694
MPH_TO_MPS = 1 / MPS_TO_MPH

# --- Simulation Parameters ---
g = 9.81  # Acceleration due to gravity (m/s^2)
court_length = 23.77  # Length of a tennis court (m)
net_height = 0.914  # Height of the net (m)
service_box_depth = 6.40 # Depth of service box from net

# Initial conditions (can be adjusted by sliders)
initial_height = 3.0  # Initial height of the ball (m) - User requested 3.0m
initial_velocity_magnitude_mph = 80.0 # User requested 80 mph
initial_velocity_magnitude_mps = initial_velocity_magnitude_mph * MPH_TO_MPS # Convert to m/s for internal use
launch_angle_degrees = 0.0  # Launch angle in degrees, adjusted to be within the new slider range [-10, 5]

# --- Calculus Explanations ---
calculus_explanation_text = (
    r"$\bf{Calculus\ in\ a\ Tennis\ Serve:}$" "\n\n"
    r"$\bf{1.\ Projectile\ Motion:}$" "\n"
    r"The ball's path is governed by:" "\n"
    r"  $x(t) = v_{0x} \cdot t$" "\n"
    r"  $y(t) = h_0 + \int_0^t v_y(\tau) d\tau = h_0 + \int_0^t (v_{0y} - g\tau) d\tau$" "\n"
    r"       $= h_0 + v_{0y}t - \frac{1}{2}gt^2$" "\n"
    r"Where:" "\n"
    r"  $v_{0x} = v_0 \cos(\theta)$ (initial horizontal velocity)" "\n"
    r"  $v_{0y} = v_0 \sin(\theta)$ (initial vertical velocity)" "\n"
    r"  $h_0$ = initial height, $g$ = gravity" "\n\n"
    r"$\bf{2.\ Velocity\ (Derivative\ of\ Position):}$" "\n"
    r"Instantaneous velocity components are found by differentiating position:" "\n"
    r"  $v_x(t) = \frac{dx}{dt} = v_{0x}$ (constant horizontal velocity, ignoring air resistance)" "\n"
    r"  $v_y(t) = \frac{dy}{dt} = v_{0y} - gt$ (vertical velocity changes due to gravity)" "\n"
    r"The speed is $\sqrt{v_x(t)^2 + v_y(t)^2}$." "\n\n"
    r"$\bf{3.\ Acceleration\ (Derivative\ of\ Velocity):}$" "\n"
    r"  $a_x(t) = \frac{dv_x}{dt} = 0$" "\n"
    r"  $a_y(t) = \frac{dv_y}{dt} = -g$ (constant downward acceleration due to gravity)" "\n\n"
    r"$\bf{Factors\ for\ a\ FAST\ Serve:}$" "\n"
    r"$\bf{A.\ Maximize\ Initial\ Velocity\ (v_0):}$" "\n"
    r"  - $\bf{Racket\ Head\ Speed:}$ The primary factor. More racket speed means more kinetic energy" "\n"
    r"    transferred to the ball. This relates to the Work-Energy Theorem:" "\n"
    r"    $W_{racket} = \Delta KE_{ball} = \frac{1}{2}mv_{ball}^2$. More work (force over distance of swing)" "\n"
    r"    means higher $v_0$." "\n"
    r"  - $\bf{Efficient\ Kinetic\ Chain:}$ Using legs, core, shoulder, arm, and wrist sequentially" "\n"
    r"    maximizes the force applied and thus the work done on the ball." "\n"
    r"  - $\bf{Sweet\ Spot\ Impact:}$ Maximizes energy transfer. Off-center hits lose energy." "\n\n"
    r"$\bf{B.\ Optimal\ Launch\ Angle\ (\theta):}$" "\n"
    r"  - Needs to be high enough to clear the net but not too high to go long." "\n"
    r"  - For a flat, fast serve, the angle is relatively small." "\n"
    r"  - The derivative $\frac{dy}{dx}$ gives the slope of the trajectory. We need $y(x_{net}) > \text{net height}$" "\n"
    r"    and $y(x_{service line}) \approx 0$ and $x_{service line} < \text{service box end}$." "\n\n"
    r"$\bf{C.\ Impact\ Point\ (h_0):}$" "\n"
    r"  - Hitting the ball at a higher point allows for a flatter trajectory (smaller $\theta$)" "\n"
    r"    while still clearing the net, which is generally faster."
)


# --- Setup the Figure and Axes ---
fig, ax = plt.subplots(figsize=(16, 10)) # Increased figure width to accommodate text
# Adjust subplot parameters:
# left: space on the left of the plot
# bottom: space below the plot (for sliders)
# right: space on the right of the plot (where the text box will go)
# top: space above the plot
VERTICAL_SHIFT = 0.45# Define a general vertical shift for the left column elements
plt.subplots_adjust(left=0.12, bottom=0.52 + VERTICAL_SHIFT, right=0.58, top=0.99, wspace=0.3)

# --- Initialize Text Objects here, after fig is created ---
# Adjusted y-positions for all text and interactive elements to make space for the embedded chart

VERTICAL_SHIFT = 0.04 # Define a general vertical shift for the left column elements

# Main plot area adjustment
# Original bottom was 0.52. New bottom = 0.52 + VERTICAL_SHIFT
plt.subplots_adjust(left=0.12, bottom=0.52 + VERTICAL_SHIFT, right=0.58, top=0.99, wspace=0.3)

# --- Create Axes for the new embedded chart (bottom left) ---
# Dimensions: [left, bottom, width, height]
# Original bottom was 0.03. New bottom = 0.03 + VERTICAL_SHIFT
ax_embedded_chart = fig.add_axes([0.12, 0.03 + VERTICAL_SHIFT, 0.46, 0.18]) 

# --- UI Elements Repositioning ---
# Original Y values are shifted up by VERTICAL_SHIFT
AX_BUTTON_Y = 0.27 + VERTICAL_SHIFT
ANGLE_RANGE_TEXT_Y = 0.32 + VERTICAL_SHIFT
AX_H0_Y = 0.35 + VERTICAL_SHIFT
AX_ANGLE_Y = 0.39 + VERTICAL_SHIFT
AX_V0_Y = 0.43 + VERTICAL_SHIFT
STATUS_TEXT_Y = 0.47 + VERTICAL_SHIFT
UI_TOP_ALIGNMENT_Y = 0.90 # This moves the calculus text box down

angle_range_text_obj = fig.text(0.15, ANGLE_RANGE_TEXT_Y, "", fontsize=9, color='green', zorder=5)
hover_tooltip_text = fig.text(0, 0, "", visible=False, fontsize=8, zorder=10,
                              ha='left', va='bottom',
                              bbox=dict(boxstyle='round,pad=0.3', fc='lemonchiffon', ec='black', alpha=0.85))
status_text_obj = fig.text(0.15, STATUS_TEXT_Y, "Adjust sliders and click 'Serve!'", fontsize=10, zorder=5)


# Draw court elements
ax.plot([0, court_length / 2], [0, 0], 'k-', lw=1) # Baseline to net (server side)
ax.plot([court_length / 2, court_length / 2], [0, net_height], 'r-', lw=2)  # Net post
service_box_marker = Rectangle((court_length / 2, -0.01), service_box_depth, 0.02, color='cornflowerblue', alpha=0.4)
ax.add_patch(service_box_marker)
ax.plot([0,0],[0, initial_height + 1], 'gray', lw=0.5) # For y-axis reference

# Mark service box end on x-axis with a circle
service_box_end_x = court_length / 2 + service_box_depth
ax.plot([service_box_end_x], [0], 'bo', markersize=5) # Small blue circle, removed label

# Annotate the service box end marker
ax.annotate('Service Box End', 
            xy=(service_box_end_x, 0), 
            xytext=(service_box_end_x, -1.0), # Position text lower, making arrow longer
            textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='blue'),
            fontsize=8, color='blue',
            bbox=dict(boxstyle='round,pad=0.3', fc='aliceblue', ec='grey', alpha=0.9),
            horizontalalignment='center', verticalalignment='top')

# Player representation (simple stick figure)
player_x = -0.5
player_body, = ax.plot([player_x, player_x], [0, initial_height - 0.5], 'ko-', lw=2, markersize=5)
player_arm, = ax.plot([player_x, player_x + 0.3], [initial_height - 0.5, initial_height], 'ko-', lw=2, markersize=3)


# Ball and trajectory plot
ball, = ax.plot([], [], 'o', color='lime', markersize=8, label="Tennis Ball")
trajectory, = ax.plot([], [], '--', color='gray', lw=1, label="Trajectory")
max_height_point, = ax.plot([],[], 'ro', markersize=5, label="Max Height")
landing_point_marker, = ax.plot([], [], 'x', color='purple', markersize=10, mew=2, label="Landing Point")


ax.set_xlabel("Distance (m)")
ax.set_ylabel("Height (m)")
ax.set_xlim(-1, court_length / 2 + service_box_depth + 2) # Show a bit beyond the service box
ax.set_ylim(0, initial_height + 2)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend(loc='upper right')

# Add the calculus explanation text box
# Position it to the right of the plot, aligning its top with the main plot's top
# Made narrower by increasing x-coordinate from 0.62 to 0.64 to elongate it vertically.
props = dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.4)
fig.text(0.62, UI_TOP_ALIGNMENT_Y, calculus_explanation_text, transform=fig.transFigure, fontsize=9,
         verticalalignment='top', bbox=props, wrap=True)

# --- Function to plot angle bounds vs. velocity (modified to plot on target_ax) ---
def plot_angle_bounds_vs_velocity(target_ax, current_h0):
    velocities_mph = np.linspace(50, 100, 51)
    
    lower_bounds_theta = []
    upper_bounds_theta = []
    plotable_velocities_mph = [] 

    # print("\nGenerating data for 'Angle Bounds vs. Velocity' chart...") # Optional: less verbose for embedded
    for v_mph in velocities_mph:
        min_theta, max_theta = find_valid_angle_range(v_mph, current_h0)
        if min_theta is not None and max_theta is not None:
            lower_bounds_theta.append(min_theta)
            upper_bounds_theta.append(max_theta)
            plotable_velocities_mph.append(v_mph)

    if not plotable_velocities_mph:
        target_ax.text(0.5, 0.5, f"No valid angle ranges found for 50-100 mph at {current_h0:.1f}m height.",
                       horizontalalignment='center', verticalalignment='center', transform=target_ax.transAxes)
        # print("No valid angle ranges found for any velocity. Cannot generate chart.") # Optional
        return
    
    target_ax.plot(plotable_velocities_mph, lower_bounds_theta, 'bo-', markersize=4, label='Min Valid θ') # Shorter labels
    target_ax.plot(plotable_velocities_mph, upper_bounds_theta, 'ro-', markersize=4, label='Max Valid θ') # Shorter labels
    target_ax.fill_between(plotable_velocities_mph, lower_bounds_theta, upper_bounds_theta, color='green', alpha=0.3, label='Valid Range')
    
    target_ax.set_xlabel("Initial Velocity (mph)", fontsize=9)
    target_ax.set_ylabel("Launch Angle θ (deg)", fontsize=9)
    target_ax.set_title(f"Valid Angle Range vs. Velocity (H₀={current_h0:.1f}m)", fontsize=10) # Shorter title
    target_ax.legend(loc='best', fontsize='small')
    target_ax.grid(True, linestyle='--', alpha=0.6)
    target_ax.tick_params(axis='x', labelsize=8)
    target_ax.tick_params(axis='y', labelsize=8)
    
    # print("'Angle Bounds vs. Velocity' chart generated.") # Optional

# --- Restored Function Definitions ---
def find_valid_angle_range(v0, h0, min_angle=-10, max_angle=5, step=0.1):
    valid_angles = []
    # v0 is passed in from the slider, which is in mph. Convert to m/s for get_trajectory.
    v0_mps = v0 * MPH_TO_MPS

    # Iterate with a finer step for better accuracy of the range
    for angle_deg in np.arange(min_angle, max_angle + step, step):
        # We only need serve_in status from get_trajectory
        results = get_trajectory(v0_mps, angle_deg, h0) # Use v0_mps here
        serve_in_status = results[6] # 7th element is serve_in
        if serve_in_status:
            valid_angles.append(angle_deg)

    if not valid_angles:
        return None, None
    else:
        return min(valid_angles), max(valid_angles)

def update_angle_range_display(val=None): # val is passed by slider on_changed
    global angle_range_text_obj
    # Ensure the text object is created and sliders are initialized before trying to set its text
    if angle_range_text_obj is None: 
        # print("Debug: angle_range_text_obj is None in update_angle_range_display")
        return
    if not hasattr(slider_v0, 'val') or not hasattr(slider_h0, 'val') or not hasattr(slider_angle, 'val'):
        # print("Debug: Sliders not ready in update_angle_range_display")
        return

    v0 = slider_v0.val
    h0 = slider_h0.val
    min_a, max_a = find_valid_angle_range(v0, h0)
    if min_a is not None and max_a is not None:
        angle_range_text_obj.set_text(f"Good Serve Angle Range: [{min_a:.1f}°, {max_a:.1f}°]")
    else:
        angle_range_text_obj.set_text("Good Serve Angle Range: None")

    # Update the embedded chart when h0 changes
    if ax_embedded_chart is not None: # Ensure the axes object exists
        ax_embedded_chart.clear() # Clear the previous plot
        plot_angle_bounds_vs_velocity(ax_embedded_chart, h0) # Redraw with new h0

    if fig.canvas: # Ensure canvas exists
        fig.canvas.draw_idle()

def on_slider_change(val):
    # This function can be used if you want live updates as sliders change,
    # but for a distinct "Serve!" button, we call run_animation from the button click.
    # For now, it's a placeholder for the angle slider, which doesn't trigger range recalculation.
    pass
# --- End of Restored Function Definitions ---

# --- Animation and Calculation Functions ---
line_ani = None # Global variable for the animation

def get_trajectory(v0, theta_deg, h0):
    theta_rad = np.deg2rad(theta_deg)
    v0x = v0 * np.cos(theta_rad)
    v0y = v0 * np.sin(theta_rad)

    # Time to reach net (if it does)
    t_at_net = (court_length / 2) / v0x if v0x > 0 else float('inf')
    y_at_net = h0 + v0y * t_at_net - 0.5 * g * t_at_net**2

    # Time to reach max height: v_y(t) = v0y - gt = 0 => t_max_h = v0y / g
    t_max_height = v0y / g if g > 0 else 0
    if t_max_height < 0: t_max_height = 0 # if launched downwards
    x_max_height = v0x * t_max_height
    y_max_height = h0 + v0y * t_max_height - 0.5 * g * t_max_height**2

    # Time to hit ground (y(t) = 0): h0 + v0y*t - 0.5*g*t^2 = 0
    # Using quadratic formula: t = (-v0y +/- sqrt(v0y^2 - 4*(-0.5g)*h0)) / (2*(-0.5g))
    # t = (-v0y +/- sqrt(v0y^2 + 2*g*h0)) / (-g)
    # We take the positive, larger root for time after launch.
    discriminant = v0y**2 + 2 * g * h0
    if discriminant < 0: # Should not happen if h0 >=0
        t_ground = float('inf')
    else:
        t_ground1 = (-v0y + np.sqrt(discriminant)) / (-g) if g!=0 else float('inf')
        t_ground2 = (-v0y - np.sqrt(discriminant)) / (-g) if g!=0 else float('inf')
        t_ground = max(t_ground1, t_ground2) if g!=0 else float('inf')
        if t_ground < 0 : # if somehow calculated a time before launch
             # This happens if launched downwards from ground, pick the other root or handle
             t_ground = (-v0y + np.sqrt(discriminant)) / (-g) if g!=0 and (-v0y + np.sqrt(discriminant)) > 0 else float('inf')
             if t_ground < 0: # Still negative, e.g. launched into ground
                 t_ground = 0


    x_ground = v0x * t_ground

    # Determine if serve is "in"
    serve_in = False
    ball_clears_net = y_at_net > net_height if t_at_net < t_ground else False
    if ball_clears_net:
        # Check if it lands in the service box
        # The service box starts at court_length/2 and ends at court_length/2 + service_box_depth
        if x_ground > court_length / 2 and x_ground < (court_length / 2 + service_box_depth):
            serve_in = True

    # Generate points for trajectory line until it hits ground or goes too far
    t_values = np.linspace(0, t_ground if t_ground != float('inf') and t_ground > 0 else 5, 200) # Max 5s or t_ground
    x_traj = v0x * t_values
    y_traj = h0 + v0y * t_values - 0.5 * g * t_values**2

    # Filter out points below ground for the trajectory line
    valid_indices = y_traj >= -0.01 # allow slightly below zero for landing point visibility
    x_traj = x_traj[valid_indices]
    y_traj = y_traj[valid_indices]


    return (x_traj, y_traj,
            x_max_height if t_max_height > 0 and t_max_height < t_ground else None,
            y_max_height if t_max_height > 0 and t_max_height < t_ground else None,
            x_ground if t_ground > 0 else None,
            0 if t_ground > 0 else None, # y_ground is 0
            serve_in, y_at_net, t_at_net, t_ground)


def update(frame, x_traj, y_traj):
    # Animate the ball along the calculated trajectory
    if frame < len(x_traj):
        ball.set_data([x_traj[frame]], [y_traj[frame]]) # Wrap in lists
    else: # Keep ball at the end if animation frames exceed trajectory points
        ball.set_data([x_traj[-1]], [y_traj[-1]]) # Wrap in lists
    return ball,

def run_animation():
    global line_ani # To control the animation object

    # Stop any existing animation robustly
    if line_ani is not None:
        old_animation = line_ani # Keep a reference to the object
        line_ani = None          # Clear global reference immediately

        # Stop the timer if it exists and is running
        if hasattr(old_animation, 'event_source') and old_animation.event_source is not None:
            try:
                old_animation.event_source.stop()
            except Exception:
                pass  # Ignore errors, timer might be already stopped

        # Disconnect the _first_draw callback. Uses the global `fig` object's canvas.
        if hasattr(old_animation, '_first_draw_id') and old_animation._first_draw_id is not None:
            try:
                # Ensure fig and fig.canvas are valid before trying to disconnect
                if fig is not None and hasattr(fig, 'canvas') and fig.canvas is not None and \
                   hasattr(fig.canvas, 'mpl_disconnect'):
                    fig.canvas.mpl_disconnect(old_animation._first_draw_id)
            except Exception:
                pass # Ignore errors during disconnect
            # Mark as None even if disconnect failed or attribute doesn't exist on old_animation post-check
            if hasattr(old_animation, '_first_draw_id'):
                 old_animation._first_draw_id = None

        # Signal to the animation's internal logic that it should stop.
        # FuncAnimation checks `self._fig is None` in many places.
        if hasattr(old_animation, '_fig'):
            old_animation._fig = None
        
        # Also clear its own event source reference
        if hasattr(old_animation, 'event_source'):
            old_animation.event_source = None

    v0 = slider_v0.val
    theta_deg = slider_angle.val
    h0 = slider_h0.val

    # Convert v0 from mph (from slider) to m/s for calculations
    v0_mph = v0
    v0_mps = v0_mph * MPH_TO_MPS

    # Update player arm position based on h0 (simplified)
    player_body.set_ydata([0, h0 - 0.5])
    player_arm.set_data([player_x, player_x + 0.3], [h0-0.5, h0])
    ax.set_ylim(0, max(4, h0 + 2)) # Adjust y_lim if h0 is high

    (x_data, y_data,
    x_max_h, y_max_h,
    x_land, y_land,
    is_in, y_net_clearance, t_net, t_land_actual) = get_trajectory(v0_mps, theta_deg, h0)

    trajectory.set_data(x_data, y_data)
    ball.set_data([], []) # Clear ball for new animation

    if x_max_h is not None and y_max_h is not None:
        max_height_point.set_data([x_max_h], [y_max_h]) # Wrapped in lists
        max_height_point.set_label(f"Max H: ({x_max_h:.1f}, {y_max_h:.1f})m")
    else:
        max_height_point.set_data([],[])
        max_height_point.set_label("Max Height")

    if x_land is not None and y_land is not None:
        landing_point_marker.set_data([x_land], [y_land]) # Wrapped in lists
        status = "IN!" if is_in else "OUT!"
        landing_point_marker.set_label(f"Landing: {x_land:.1f}m ({status})")

        # Display velocities at landing (using derivatives)
        v_land_x = v0_mps * np.cos(np.deg2rad(theta_deg))
        v_land_y = v0_mps * np.sin(np.deg2rad(theta_deg)) - g * t_land_actual
        speed_land_mps = np.sqrt(v_land_x**2 + v_land_y**2)
        speed_land_mph = speed_land_mps * MPS_TO_MPH
        ax.set_title(f"Serve Status: {status} | Landing Speed: {speed_land_mph:.1f} mph", color='green' if is_in else 'red')

    else:
        landing_point_marker.set_data([],[])
        landing_point_marker.set_label("Landing Point")
        ax.set_title("Serve Status: (ball does not land in range)")


    # Update clearance info
    clearance_text = (f"Ball height at net: {y_net_clearance:.2f}m. "
                      f"Net height: {net_height:.2f}m.")
    if t_net > t_land_actual and t_land_actual > 0: # Hits ground before net
        clearance_text = "Ball hits ground before reaching net."
    elif y_net_clearance == float('inf') or x_data[-1] < court_length/2 :
        clearance_text = "Ball does not reach net."

    status_text_obj.set_text(clearance_text)


    ax.legend(loc='upper right', fontsize='small')

    # Create and start the animation
    if len(x_data) > 0:
        num_frames = len(x_data)
        # Ensure num_frames * v0_mps is not zero for division, and interval is an int
        # Use v0_mps for interval calculation as it relates to physical travel time
        denominator = num_frames * v0_mps 
        interval_ms = 50 # Default interval if denominator is zero
        if denominator > 0: 
            interval_ms = int(50000 / denominator) # 5000 * 10 / (num_frames * v0_mps)

        line_ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                           fargs=(x_data, y_data), interval=max(1, interval_ms),
                                           blit=True, repeat=False)
    # If len(x_data) == 0, line_ani remains None (set after stopping old one)
    
    fig.canvas.draw_idle()


# --- Sliders and Button ---
axcolor = 'lightgoldenrodyellow'
# Adjust slider positions to be under the plot area
ax_v0 = plt.axes([0.15, AX_V0_Y, 0.4, 0.03], facecolor=axcolor)
ax_angle = plt.axes([0.15, AX_ANGLE_Y, 0.4, 0.03], facecolor=axcolor)
ax_h0 = plt.axes([0.15, AX_H0_Y, 0.4, 0.03], facecolor=axcolor)
ax_button = plt.axes([0.41, AX_BUTTON_Y, 0.09, 0.04]) # Reset button

slider_v0 = Slider(ax_v0, r'Initial Velocity ($v_0$) mph', 5.0 * MPS_TO_MPH, 70.0 * MPS_TO_MPH, valinit=initial_velocity_magnitude_mph)
slider_angle = Slider(ax_angle, r'Launch Angle ($	heta$) degrees', -10.0, 5.0, valinit=launch_angle_degrees)
slider_h0 = Slider(ax_h0, r'Initial Height ($h_0$) m', 1.5, 3.5, valinit=initial_height)
btn_animate = Button(ax_button, 'Serve!', color='lightblue', hovercolor='0.975')

slider_v0.on_changed(update_angle_range_display)
slider_angle.on_changed(on_slider_change) # Angle slider does not change the valid range itself
slider_h0.on_changed(update_angle_range_display)
btn_animate.on_clicked(lambda event: run_animation())

# Initial plot setup
run_animation() # This will also create the plot and sliders

# Now that sliders are created by run_animation (which calls get_trajectory etc., and sets up UI),
# and angle_range_text_obj is initialized above, we can update the display.
update_angle_range_display() 

# --- Mouse Hover Event for Sliders ---
def on_hover(event):
    global hover_tooltip_text 
    if hover_tooltip_text is None: return

    active_slider_text = None
    # Check if mouse is over any of the slider axes and event.xdata is available
    if event.inaxes == ax_v0 and event.xdata is not None:
        value_at_cursor_mph = np.clip(event.xdata, slider_v0.valmin, slider_v0.valmax)
        active_slider_text = f"Velocity: {value_at_cursor_mph:.1f} mph"
    elif event.inaxes == ax_angle and event.xdata is not None:
        value_at_cursor = np.clip(event.xdata, slider_angle.valmin, slider_angle.valmax)
        active_slider_text = f"Angle: {value_at_cursor:.1f}°"
    elif event.inaxes == ax_h0 and event.xdata is not None:
        value_at_cursor = np.clip(event.xdata, slider_h0.valmin, slider_h0.valmax)
        active_slider_text = f"Height: {value_at_cursor:.1f} m"

    if active_slider_text and event.x is not None and event.y is not None:
        fig_width_px = fig.get_size_inches()[0] * fig.dpi
        fig_height_px = fig.get_size_inches()[1] * fig.dpi

        if fig_width_px == 0 or fig_height_px == 0: # Fig not ready
            hover_tooltip_text.set_visible(False)
            if fig.canvas:
                 fig.canvas.draw_idle()
            return

        # Convert mouse display coordinates to figure fraction coordinates
        fx = event.x / fig_width_px
        fy = event.y / fig_height_px

        hover_tooltip_text.set_text(active_slider_text)
        # Position slightly above and to the right of the cursor
        hover_tooltip_text.set_position((fx + 0.015, fy + 0.015)) 
        hover_tooltip_text.set_visible(True)
    else:
        hover_tooltip_text.set_visible(False)
    
    if fig.canvas:
        fig.canvas.draw_idle()

# Connect the hover event to the figure
fig.canvas.mpl_connect('motion_notify_event', on_hover)

# Generate the new chart on the ax_embedded_chart Axes object
plot_angle_bounds_vs_velocity(ax_embedded_chart, initial_height) # Pass initial_height

plt.show()