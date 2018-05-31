
type 'a nlist =
  | Cons of 'a * 'a nlist
  | Nil

let rec f x = match x with
  | 1000 -> print_int 1000
  | x -> print_int x; f (x + 1)

let rec sum lst = match lst with
  | [] -> 0
  | hd :: tl -> hd + (sum tl)

let rec rev lst = match lst with
  | [] -> []
  | [x] -> x :: []
  | hd :: tl -> (rev tl) @ [hd]

let make_list (max: int) =
  let rec f (x: int list) : int list = match x with
    | [] -> f [0]
    | x :: tl when x < (max-1) -> f (x + 1 :: x :: tl)
    | x :: tl -> x :: tl
  in
  f []

let rec elem lst e = match lst with
  | [] -> false
  | hd :: tl ->
    if hd = e then
      true
    else
      elem tl e
