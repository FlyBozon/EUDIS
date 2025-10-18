Import("env")

# Ensure the model folder is in include path (so we can keep huge headers out of VCS sync limits)
model_dir = env.subst("$PROJECT_DIR") + "/model"
cpppath = env.get("CPPPATH", [])
if model_dir not in cpppath:
    cpppath.append(model_dir)
    env.Replace(CPPPATH=cpppath)
print("[PIO] Added model include path:", model_dir)
