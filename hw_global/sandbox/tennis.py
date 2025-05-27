import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

# --- Simulation Parameters ---
g = 9.81  # Acceleration due to gravity (m/s^2)
court_length = 23.77  # Length of a tennis court (m)
net_height = 0.914  # Height of the net (m)
service_box_depth = 6.40 # Depth of service box from net

# Initial conditions (can be adjusted by sliders)
initial_height = 2.5  # Initial height of the ball (m) - typical for a serve
initial_velocity_magnitude = 30  # Initial velocity of the ball (m/s) - about 67 mph
launch_angle_degrees = 10  # Launch angle in degrees

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
fig, ax = plt.subplots(figsize=(16, 7)) # Increased figure width to accommodate text
# Adjust subplot parameters:
# left: space on the left of the plot
# bottom: space below the plot (for sliders)
# right: space on the right of the plot (where the text box will go)
# top: space above the plot
plt.subplots_adjust(left=0.12, bottom=0.30, right=0.58, top=0.97, wspace=0.3)

# Draw court elements
ax.plot([0, court_length / 2], [0, 0], 'k-', lw=1) # Baseline to net (server side)
ax.plot([court_length / 2, court_length / 2], [0, net_height], 'r-', lw=2)  # Net post
ax.plot([court_length / 2, court_length / 2 + service_box_depth], [0,0], 'b-', lw=1) # Service line
ax.plot([0,0],[0, initial_height + 1], 'gray', lw=0.5) # For y-axis reference

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
# Position it to the right of the plot
props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.4)
fig.text(0.62, 0.05, calculus_explanation_text, transform=fig.transFigure, fontsize=8,
         verticalalignment='bottom', bbox=props, wrap=True)


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
        ball.set_data(x_traj[frame], y_traj[frame])
    else: # Keep ball at the end if animation frames exceed trajectory points
        ball.set_data(x_traj[-1], y_traj[-1])
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

    # Update player arm position based on h0 (simplified)
    player_body.set_ydata([0, h0 - 0.5])
    player_arm.set_data([player_x, player_x + 0.3], [h0-0.5, h0])
    ax.set_ylim(0, max(4, h0 + 2)) # Adjust y_lim if h0 is high

    (x_data, y_data,
    x_max_h, y_max_h,
    x_land, y_land,
    is_in, y_net_clearance, t_net, t_land_actual) = get_trajectory(v0, theta_deg, h0)

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
        v_land_x = v0 * np.cos(np.deg2rad(theta_deg))
        v_land_y = v0 * np.sin(np.deg2rad(theta_deg)) - g * t_land_actual
        speed_land = np.sqrt(v_land_x**2 + v_land_y**2)
        ax.set_title(f"Serve Status: {status} | Landing Speed: {speed_land:.1f} m/s", color='green' if is_in else 'red')

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
        # Ensure num_frames * v0 is not zero for division, and interval is an int
        denominator = num_frames * v0
        interval_ms = 50 # Default interval if denominator is zero (should not happen with v0 > 0)
        if denominator > 0: # v0 slider starts at 5.0, num_frames > 0 here
            interval_ms = int(50000 / denominator) # 5000 * 10 / (num_frames * v0)

        line_ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                           fargs=(x_data, y_data), interval=max(1, interval_ms),
                                           blit=True, repeat=False)
    # If len(x_data) == 0, line_ani remains None (set after stopping old one)
    
    fig.canvas.draw_idle()


# --- Sliders and Button ---
axcolor = 'lightgoldenrodyellow'
# Adjust slider positions to be under the plot area
ax_v0 = plt.axes([0.15, 0.20, 0.4, 0.03], facecolor=axcolor)
ax_angle = plt.axes([0.15, 0.15, 0.4, 0.03], facecolor=axcolor)
ax_h0 = plt.axes([0.15, 0.10, 0.4, 0.03], facecolor=axcolor)
ax_button = plt.axes([0.41, 0.03, 0.09, 0.04]) # Reset button

slider_v0 = Slider(ax_v0, r'Initial Velocity ($v_0$) m/s', 5.0, 70.0, valinit=initial_velocity_magnitude)
slider_angle = Slider(ax_angle, r'Launch Angle ($	heta$) degrees', -10.0, 45.0, valinit=launch_angle_degrees)
slider_h0 = Slider(ax_h0, r'Initial Height ($h_0$) m', 1.5, 3.5, valinit=initial_height)
btn_animate = Button(ax_button, 'Serve!', color='lightblue', hovercolor='0.975')

# Status text below sliders, aligned with sliders
status_text_obj = fig.text(0.15, 0.25, "Adjust sliders and click 'Serve!'", fontsize=10)


def on_slider_change(val):
    # This function can be used if you want live updates as sliders change,
    # but for a distinct "Serve!" button, we call run_animation from the button click.
    # For now, we'll just let the button trigger the animation.
    pass

slider_v0.on_changed(on_slider_change)
slider_angle.on_changed(on_slider_change)
slider_h0.on_changed(on_slider_change)
btn_animate.on_clicked(lambda event: run_animation())

# Initial plot
run_animation()
plt.show()