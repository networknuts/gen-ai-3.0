user_input = """
my name is michael and my email is robby@networknuts.net.
write an email to my boss requesting 7 days off
due to personal reasons.
"""

llm_output = """
here is your email which you can copy and paste:

to: boss@networknuts.net
subject: requesting x days off
body:
from: aryan@networknuts.net
"""

import re 

result_input = re.search(r"\w+@\w+\.\w+",user_input)
result_output = re.search(r"[A-Za-z0-9_]+@[A-Za-z0-9]+\.[a-zA-Z0-9]+",llm_output)

result_redacted = re.sub(r"\w+@\w+\.\w+",r"[REDACTED]",user_input)
print(result_redacted)
